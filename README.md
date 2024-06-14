# NARes Evaluation

The Evaluation code of [**NARes**](https://github.com/zhichao-lu/arch-dataset-adv-robustness)

## Preparation

1. Put dataset (CIFAR-10, CIFAR-10-C) to `datasets/`:

    ```
    datasets
    ├── cifar10
    │   └── cifar-10-batches-py
    └── cifar10c
    ```

2. Put model weights and config file to `models_home/`:

    ```
    models_home
    └── arch_170
        ├── arch_170.yaml
        └── checkpoints
            └── arch_170_best.pth
    ```

3. Install packages from `requirements.txt`

## Attack Robustness Evaluation

```shell
model=arch_2333

#Base Linf
python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0

#Base L2
python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0 --norm L2

#AA-Linf
python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0 --attack-choice AA --aa-type Compact

#AA-L2
python ./eval.py --model-path ./models_home --log-path ./attack_log --load-best-model --model ${model} --seed 0 --attack-choice AA --aa-type Compact --norm L2
```

Evalute models by a text file with arch_id list:
```shell
# run on GPU0 with Base-Linf and seed 42
./run.sh 0 arch_list.txt base Linf 42

# run on GPU3 with AA-Linf and seed 22
./run.sh 3 arch_list.txt aa Linf 22
```
## Corruption Robustness Evaluation

```shell
python ./eval_corruption_level.py --model-path ./models_home --log-path ./attack_log --load-best-model --model arch_2333
```

<!-- ## Get Test Loss

```shell
python ./eval_test_loss.py --model-path ./models_home --log-path ./attack_log --load-best-model --model arch_11451
``` -->

# Acknowledgement

* [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)

* [auto-attack](https://github.com/fra31/auto-attack)

* [CIFAR-10-C](https://github.com/hendrycks/robustness)