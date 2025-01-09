#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
jsq=/mnt/nvme1/wangzining/llm-jsq
export PYTHONPATH=$jsq:$PYTHONPATH


task_name=test



python ${jsq}/main.py \
--model /mnt/nvme1/models/llama2/llama2-7b \
--weight_quant per_channel \
--act_quant per_token \
--a_bits 6 \
--w_bits 6 \
--pruning_method wanda \
--rho 2.1 \
--multigpu \
--sparsity_ratio 0.43 \
--sparsity_type unstructured \
--tasks piqa,boolq,hellaswag \


# python ${jsq}/main.py \


# --model /mnt/nvme1/models/llama2/llama2-7b \
# --pruning_method wanda \
# --rho 2.1 \
# --sparsity_ratio 0.5 \
# --sparsity_type unstructured \
# --tasks wikitext \
