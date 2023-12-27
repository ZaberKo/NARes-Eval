# r-nas-attack

## Evaluation

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
