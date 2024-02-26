# r-nas-attack

## Attack Robustness Evaluation

Examples:

Put model weights and config file to `models_home`:

```
models_home
└── arch_170
    ├── arch_170.yaml
    └── checkpoints
        └── arch_170_best.pth
```

```shell
model=arch_11451

#Base Linf
python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0

#Base L2
python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0 --norm L2

#AA-Linf
python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0 --attack-choice AA --aa-type Compact

#AA-L2
python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0 --attack-choice AA --aa-type Compact --norm L2
```

Evalute models by a file with arch_id list:
```shell
# run on GPU0 with Base Linf
./run.sh 0 arch_list.txt base Linf

# run on GPU3 with AA Linf
./run.sh 3 arch_list.txt aa Linf
```
## Corruption Robustness Evaluation

Put the CIFAR-10-C data to `datasets/cifar10c`

```shell
python ./eval_corruption.py --model-path ./models_home --log-path ./attack_log --load-best-model --model arch_11451
```

## Get Test Loss

```shell
python ./eval_test_loss.py --model-path ./models_home --log-path ./attack_log --load-best-model --model arch_11451
```