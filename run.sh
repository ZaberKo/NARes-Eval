#!/bin/bash
set -eu

GPU=${1:-0}
arch_list=${2:-arch_list.txt}
attack_type=${3:-base}
norm=${4:-Linf}
seed=${5:-0}



while IFS= read -r arch_id; do
    model=arch_$arch_id
    if [[ ${attack_type} == "base" ]]
    then
        (set +x; CUDA_VISIBLE_DEVICES=${GPU} python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed ${seed} --norm ${norm})
    elif [[ ${attack_type} == "aa" ]]
    then
        (set +x; CUDA_VISIBLE_DEVICES=${GPU} python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed ${seed} --norm ${norm} --attack-choice AA --aa-type Compact)
    elif [[ ${attack_type} == "corruption" ]]
    then
        (set +x; CUDA_VISIBLE_DEVICES=${GPU} python ./eval_corruption_level.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed ${seed})
    else
        echo "Invalid attack type: ${attack_type}"
    fi
done < ${arch_list}