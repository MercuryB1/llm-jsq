import torch
import torch.nn as nn
import functools
from tqdm import tqdm
from .utils import get_loaders, clip_matrix
from loguru import logger
from .utils import prepare_calibration_input, llama_eval, move_embed, get_blocks, get_named_linears
from .prune import auto_prune_block
from .layer_wrapper import WrappedLayer
from .smooth import smooth_layer
from .fake_quant import quantize_layer
from .autoclip import auto_clip_block, apply_clip
from .calibration import get_layer_act_scale
from .data import get_calib_dataset
from copy import deepcopy
from collections import defaultdict
import gc


@torch.no_grad()
def annealing_loop(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    
    layers = get_blocks(model)
    logger.info("loading calibdation data")
    # dataloader, testenc = get_loaders(
    #     args.calib_dataset, 
    #     nsamples=args.nsamples,
    #     seed=args.seed,
    #     seqlen=model.seqlen,
    #     tokenizer=tokenizer
    # )
    samples = get_calib_dataset(
        data=args.calib_dataset,
        tokenizer=tokenizer,
        n_samples=args.nsamples,
        seq_len=args.seqlen,
    )
    samples = torch.cat(samples, dim=0)
    logger.info("dataset loading complete")
    
    inps = []
    layer_kwargs = {}
    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    
    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            layer_kwargs['use_cache'] = False
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(samples.to(next(model.parameters()).device))
        else:
            model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    
    layers[0] = layers[0].module  # restore
    inps = inps[0]
    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()
    
    for i in tqdm(range(len(layers)), desc="Running JSQ..."):
        layer = layers[i]
        layer.cuda()
        named_linears = get_named_linears(layer)
        
         # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        
        
        layer(inps, **layer_kwargs)[0]
        
        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        
        logger.info(f"pruning layer {i}")
        auto_prune_block(args, layer, input_feat, args.sparsity_ratio, prune_n, prune_m, args.use_variant)
        
        act_scales = get_layer_act_scale(layer, inps, **layer_kwargs)
        
        logger.info(f"smoothing layer {i}")
        smooth_layer(layer, act_scales, 0.8)
        
        logger.info(f"clipping layuer {i}")
        clip_list = auto_clip_block(layer, w_bits=args.w_bits, input_feat=input_feat, n_sample_token=args.nsamples)
        
        logger.info(f"applying clip for layer {i}")
        apply_clip(layer, clip_list)

        logger.info(f"quantizing layer {i}")
        quantize_layer(layer, w_bits=args.w_bits, a_bits=args.a_bits, quantize_bmm_input=True)
        

        # update output after compression
        inps = layer(inps, **layer_kwargs)[0]
        
        del input_feat, act_scales
        layer.cpu()
        torch.cuda.empty_cache()
        
        

# def prune_and_quant(
#     args, model, device, prune_n=0, prune_m=0, inputs=None
# ):
    
#     layers = model.model.layers
#     for i in tqdm(range(len(layers)), desc="Running JSQ..."):
#         layer = layers[i]
#         layer_name = f'model.layers.{i}'
#         subset = find_layers(layer)
        
#         if '30b' in args.model and f"model.layers.{i}" in model.hf_device_map:  
#         #handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
#             dev = model.hf_device_map[f"model.layers.{i}"]
#             inputs = {key: tensor.to(dev) for key, tensor in inputs.items()}
#         act_scales = {}
#         wrapped_layers = {}
#         for name in subset:
#             wrapped_layers[name] = WrappedLayer(subset[name])
            
#         def stat_tensor(name, tensor):
#             hidden_dim = tensor.shape[-1]
#             tensor = tensor.view(-1, hidden_dim).abs().detach()
#             comming_max = torch.max(tensor, dim=0)[0].float().cpu()
#             if name in act_scales:
#                 act_scales[layer_name + '.' + name] = torch.max(act_scales[name], comming_max)
#             else:
#                 act_scales[layer_name + '.' + name] = comming_max
        
#         def add_batch(name):
#             def tmp(_, inp, out):
#                 inp = inp[0].data
#                 inp = clip_matrix(inp, args.abs, args.clip_l, clip_opts[clip_table[i]])
#                 stat_tensor(name, inp)
#                 wrapped_layers[name].add_batch(inp, out.data)
#             return tmp
        
#         handles = []
#         for name in wrapped_layers:
#             handles.append(subset[name].register_forward_hook(add_batch(name)))
        
#         for j in range(args.nsamples):
#             with torch.no_grad():
#                 inputs["outs"][j] = layer(
#                     inputs["inps"][j].unsqueeze(0), 
#                     attention_mask=inputs["attention_mask"], 
#                     position_ids=inputs["position_ids"]
#                 )[0]
                
#         for h in handles:
#             h.remove()
            
#         prune_wanda()
#         for j in range(args.nsamples):
#             with torch.no_grad():
#                 inputs["outs"][j] = layer(
#                     inputs["inps"][j].unsqueeze(0), 
#                     attention_mask=inputs["attention_mask"], 
#                     position_ids=inputs["position_ids"]
#                 )[0]
                
#         logger.info(f"smoothing layer {i}")
#         smooth_layer(layer_name, layer, act_scales, 0.8)
        
#         logger.info(f"clipping layuer {i}")
#         clip_list = auto_clip_block(layer, w_bits=args.w_bits, input_feat=inputs, n_sample_token=args.nsamples)
        
#         logger.info(f"applying clip for layer {i}")
#         apply_clip(layer, clip_list)

#         logger.info(f"quantizing layer {i}")
#         quantize_layer(layer, w_bits=args.w_bits, a_bits=args.a_bits, quantize_bmm_input=True)
#         inps, outs = outs, inps
                
        
#     logger.info('begin eval')
#     ppl = 0
#     ppl = llama_eval(model, testenc, device)
#     logger.info(f'SmoothQuant W8A8 quantized model ppl: {ppl}')
#     del model
#     torch.cuda.empty_cache()
#     return ppl