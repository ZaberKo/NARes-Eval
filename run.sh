#!/bin/bash
set -eu

GPU=${1:-0}
arch_list=${2:-arch_list.txt}
attack_type=${3:-base}
norm=${4:-Linf}


while IFS= read -r arch_id; do
    model=arch_$arch_id
    if [[ ${attack_type} == "base" ]]
    then
        (set +x; CUDA_VISIBLE_DEVICES=${GPU} python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0 --norm ${norm})
    elif [[ ${attack_type} == "aa" ]]
    then
        (set +x; CUDA_VISIBLE_DEVICES=${GPU} python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0  --norm ${norm} --attack-choice AA --aa-type Compact)
    else
        echo "Invalid attack type: ${attack_type}"
    fi
done < ${arch_list}