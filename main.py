import random
import numpy as np
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from loguru import logger
from jsq.compression.prune import check_sparsity
from jsq.compression.jsq import annealing_loop
from tqdm import tqdm
from jsq.utils.lm_eval_adaptor import LMEvalAdaptor
from jsq.utils.parellel_utils import map_layers_to_multi_gpus
from lm_eval import evaluator, tasks
import json
from datasets import load_dataset
from jsq.utils.categories import subcategories, categories


def seed_everything(seed: int):
    random.seed(seed)  # Python built-in random module
    np.random.seed(seed)  # NumPy

    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash seed (for consistent hashing)

    # PyTorch
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU (single card)
    torch.cuda.manual_seed_all(seed)  # GPU (multiple cards)

    # Ensure deterministic behavior in PyTorch (if necessary)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, tokenizer, args):
    results = {}
    lm = LMEvalAdaptor(args.model, model, tokenizer, args.batch_size)
    if args.multigpu:
        if "llama" in args.model.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
    if args.eval_ppl:
        testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
        model.seqlen = 2048
        testenc = testenc.input_ids.to(model.device)
        nsamples = testenc.numel() // model.seqlen
        nlls = []
        for i in tqdm(range(nsamples), desc="evaluating..."):
            batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                model.device
            )
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = testenc[
                :, (i * model.seqlen) : ((i + 1) * model.seqlen)
            ][:, 1:].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        logger.info(ppl.item())

        results = {"ppl": ppl.item()}
    if args.tasks is not None:
        
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        # logger.info(results)
        print(evaluator.make_table(results))
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               
    return results


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model


def build_model(args):
    model_name = args.model
    if 'glm' in model_name:
        kwargs = {'torch_dtype': torch.float16, 'device_map': 'auto', 'trust_remote_code': True}
    else:
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.seqlen = model.config.max_position_embeddings 
    return model


def build_model_and_enc(args):
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.use_cache = False
    enc = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=True
    )
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, trust_remote_code=True, **kwargs
    )
    model.eval()
    return model, enc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name or model path")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--calib_dataset",type=str,default="pileval",
        choices=["wikitext2", "ptb", "c4", "mix","pileval"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seqlen", type=int, default=2048, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true", help="eval wikitext2 ppl")
    parser.add_argument("--multigpu", action="store_true", help="use multigpu for eval")
    parser.add_argument("--tasks", default=None, type=str)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--weight_quant", type=str, default="per_channel", choices=["per_channel", "per_tensor"])
    parser.add_argument("--act_quant", type=str, default="per_token", choices=["per_token", "per_tensor"])
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5) 
    parser.add_argument("--pruning_method", type=str, default="jsq_v1",
        choices=["jsq_v1", "jsq_v2", "wanda", "magnitude"],
        help="Pruning metric selection.",
    )
    parser.add_argument("--rho", type=float, default=2, help="lambda of eq.(4)")
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--use_variant", action="store_true", help="use variant pruning")
    
    args = parser.parse_args()
    seed_everything(args.seed)
    
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
        
    logger.info(f"loading llm model {args.model}")
    model, tokenizer = build_model_and_enc(args)
    # model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    logger.info(f"use device: {device}")
    
    # annealing_loop(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    # logger.info("*"*30)
    # sparsity_ratio = check_sparsity(model)
    # logger.info(f"sparsity sanity check {sparsity_ratio:.4f}")
    # logger.info("*"*30)
    
    evaluate(model, tokenizer, args)
    
    # if args.tasks is not None:
    #     if args.tasks == "wikitext":
    #         testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    #         testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    #         model.seqlen = 2048
    #         testenc = testenc.input_ids.to(model.device)
    #         nsamples = testenc.numel() // model.seqlen
    #         model = model.eval()
    #         model.cuda()
    #         nlls = []
    #         for i in tqdm(range(nsamples), desc="evaluating..."):
    #             batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
    #                 model.device
    #             )
    #             with torch.no_grad():
    #                 lm_logits = model(batch).logits
    #             shift_logits = lm_logits[:, :-1, :].contiguous().float()
    #             shift_labels = testenc[
    #                 :, (i * model.seqlen) : ((i + 1) * model.seqlen)
    #             ][:, 1:].to(shift_logits.device)
    #             loss_fct = nn.CrossEntropyLoss()
    #             loss = loss_fct(
    #                 shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    #             )
    #             neg_log_likelihood = loss.float() * model.seqlen
    #             nlls.append(neg_log_likelihood)

    #         ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    #         logger.info(ppl.item())

    #         results = {"ppl": ppl.item()}
    #         # if args.output_path is not None:
    #         #     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    #         #     with open(args.output_path, "w") as f:
    #         #         json.dump(results, f, indent=2)
    #     else:
    #         task_names = args.tasks.split(",")
    #         model = model.eval()
    #         model.cuda()
    #         lm_eval_model = LMEvalAdaptor(args.model, model, tokenizer, args.batch_size)
    #         results = evaluator.simple_evaluate(
    #             model=lm_eval_model,
    #             tasks=task_names,
    #             batch_size=args.batch_size,
    #             no_cache=True,
    #             num_fewshot=args.num_fewshot,
    #         )

    #         print(evaluator.make_table(results))
    
    

if __name__ == "__main__":
    main()