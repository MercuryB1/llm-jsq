#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
jsq=/mnt/nvme1/wangzining/llm-jsq
export PYTHONPATH=$jsq:$PYTHONPATH


task_name=test



python ${jsq}/main.py \
--model /mnt/nvme1/models/llama2/llama2-7b \
--weight_quant per_channel \
--act_quant per_token \
--a_bits 8 \
--w_bits 8 \
--pruning_method wanda \
--rho 2.1 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--tasks wikitext \


# python ${jsq}/main.py \


# --model /mnt/nvme1/models/llama2/llama2-7b \
# --pruning_method wanda \
# --rho 2.1 \
# --sparsity_ratio 0.5 \
# --sparsity_type unstructured \
# --tasks wikitext \
