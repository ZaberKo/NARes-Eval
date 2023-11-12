docker run -it --gpus=all \
    --name nasbenchR-attack \
    --hostname nasbenchR-attack \
    -v ~/workspace/nasbenchR-attack:/root/nasbenchR-attack \
    -P \
    --shm-size=8g \
    harbor.lightcube.zaberlab.com/library/robust-nasbench:20.12-torch1.8.0-cuda11.1-py3.8 


python ./eval.py --model-path ./models_home --load-best-model --model arch_170 --seed 0 --progress-bar

python ./eval.py --model-path ./models_home --load-best-model --model arch_170 --seed 0 --norm L2 --progress-bar