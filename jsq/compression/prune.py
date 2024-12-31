import time 
import heapq 
import torch 
import torch.nn as nn 
from loguru import logger 
from .layer_wrapper import WrappedLayer
from jsq.compression.fake_quant import QuantLinear


def find_layers(module, layers=[nn.Linear, QuantLinear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        logger.info(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count) / total_params 


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0
            

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedLayer(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def auto_prune_block(args, module, input_feat, sparsity_ratio, prune_n=0, prune_m=0, use_variant=False):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }
    for name in named_linears:
        if args.pruning_method == "wanda":
            auto_prune_layer_wanda(
                named_linears[name].weight, input_feat[name], sparsity_ratio, prune_n, prune_m, use_variant
            )
        elif "jsq" in args.pruning_method:
            auto_prune_layer_jsq(
                args.pruning_method, args.rho, named_linears[name].weight, input_feat[name], sparsity_ratio, prune_n, prune_m
            )
            
            
@torch.no_grad()
def auto_prune_layer_jsq(pruning_method, rho, w, input_feat, sparsity_ratio, prune_n=0, prune_m=0):
    inp = input_feat[0]
    
    # # calculate outputs for the jsq metric
    # oup = inp.to(w.device) @ w.data.t()
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    nsamples = inp.shape[0]
    if len(inp.shape) == 3:
        inp = inp.reshape((-1, inp.shape[-1]))
    inp = inp.t()
    
    # wanda metric
    columns = w.data.shape[1]
    scaler_row = torch.zeros((columns), device=w.device)
    inp = inp.type(torch.float32).to(scaler_row.device)
    scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / nsamples
    
    # auxiliary salience
    if pruning_method == "jsq_v1":
        activation = input_feat[0].to(w.device)
        cout, cin = w.shape
    
        # 计算原始输出矩阵
        original_out = activation @ w.T  # (batch, cin) @ (cin, cout) -> (batch, cout)

        # 初始化敏感性矩阵
        ss = torch.zeros_like(w, device=w.device)  # 初始化敏感性矩阵

        # 遍历每个权重行并计算影响
        for i in range(cout):
            column_out = original_out[:, i]  # 获取当前列的原始输出

            # 计算每一列的贡献并并行处理
            contributions = activation * w[i, :].unsqueeze(0)  # (batch, cin)
            modified_column_out = column_out.unsqueeze(1) - contributions  # (batch, cin)

            # 计算每列的最大值和最小值差
            max_values, _ = torch.max(modified_column_out, dim=0)  # (cin,)
            min_values, _ = torch.min(modified_column_out, dim=0)  # (cin,)
            ss[i, :] = max_values - min_values  # 存储结果
            del contributions, modified_column_out, column_out
            
        ss[torch.isinf(ss)] = 100
    elif pruning_method == "jsq_v2":
        pass
    else:
        raise NotImplementedError(f"not supported method")
    
    
    W_metric = torch.abs(w.data) * torch.sqrt(scaler_row.reshape((1,-1))) + rho * ss
    W_mask = (torch.zeros_like(W_metric) == 1)
    
    if prune_n != 0:
        # structured n:m sparsity
        for i in range(W_metric.shape[1]):
            if i % prune_m == 0:
                tmp = W_metric[:,i:(i+prune_m)].float()
                W_mask.scatter_(1,i+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        # unstructured pruning
        indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
        W_mask.scatter_(1, indices, True)

    w.data[W_mask] = 0  ## set weights to zero 
        

def auto_prune_layer_wanda(w, input_feat, sparsity_ratio, prune_n=0, prune_m=0, use_variant=False):

    inp = input_feat[0]
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    nsamples = inp.shape[0]
    if len(inp.shape) == 3:
        inp = inp.reshape((-1, inp.shape[-1]))
    inp = inp.t()
    
    columns = w.data.shape[1]
    scaler_row = torch.zeros((columns), device=w.device)
    inp = inp.type(torch.float32).to(scaler_row.device)
    scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / nsamples
    
    W_metric = torch.abs(w.data) * torch.sqrt(scaler_row.reshape((1,-1)))
    W_mask = (torch.zeros_like(W_metric) == 1)
    
    if prune_n != 0:
        # structured n:m sparsity
        for i in range(W_metric.shape[1]):
            if i % prune_m == 0:
                tmp = W_metric[:,i:(i+prune_m)].float()
                W_mask.scatter_(1,i+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)

        if use_variant:
            # wanda variant 
            tmp_metric = torch.cumsum(sort_res[0], dim=1)
            sum_before = W_metric.sum(dim=1)

            alpha = 0.4
            alpha_hist = [0., 0.8]
            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
            while (torch.abs(cur_sparsity - sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                if cur_sparsity > sparsity_ratio:
                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                    alpha_hist[1] = alpha
                else:
                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                    alpha_hist[0] = alpha

                alpha = alpha_new 
                W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
        else:
            # unstructured pruning
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

    w.data[W_mask] = 0  ## set weights to zero 
     