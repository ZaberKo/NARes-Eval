import mlconfig
import argparse
import util
import dataset
import time
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchprofile import profile_macs
from pathlib import Path

from collections import defaultdict
import tqdm

# used to register the models
import models

from base_attackers import Evaluator
from auto_attack.autoattack import AutoAttack
mlconfig.register(dataset.DatasetGenerator)


def setup_args():
    parser = argparse.ArgumentParser(
        description='Adversarial Attack Evaluate: Linf/L2-version of FGSM, PGD20, CW40, AutoAttack with epsilon@8/255;')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default="arch_001")
    parser.add_argument('--model-path', type=str, default='models_home')
    parser.add_argument('--log-path', type=str)
    parser.add_argument('--load-best-model',
                        action='store_true', default=False)
    parser.add_argument('--progress-bar', action='store_true')

    args = parser.parse_args()

    return args


def get_ratio(metric: util.AverageMeter):
    return f'{round(metric.sum)}/{round(metric.count)}'



def test_loss_eval(data_loader, model, progress_bar=False):
    metrics = defaultdict(util.AverageMeter)
    model.eval()
    if progress_bar:
        _data_loader = tqdm.tqdm(data_loader["test_dataset"])
    else:
        _data_loader = data_loader["test_dataset"]

    for i, (images, labels) in enumerate(_data_loader):
        images, labels = images.to(device), labels.to(device)
        bs = images.size(0)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        metrics['test_loss'].update(loss.item(), n=bs)

    return metrics



def main():
    # Load Search Version Genotype
    model = config.model().to(args.device)
    # logger.info(model)

    # Setup train_statedict
    data_loader = config.dataset().getDataLoader()

    # load trained weights

    checkpoint = util.load_model(
        filename=checkpoint_file, model=model, optimizer=None, alpha_optimizer=None, scheduler=None)
    logger.info(f"File {checkpoint_file} loaded!")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model = torch.nn.DataParallel(model).to(args.device)

    
    result_dict = test_loss_eval(data_loader, model, args.progress_bar)
    logger.info(f"Test Loss: {result_dict['test_loss'].avg:.4f}")


if __name__ == '__main__':
    args = setup_args()

    # get the corresponding directory
    model_path = Path(args.model_path)/args.model

    if args.log_path is None:
        log_path = model_path
    else:
        log_path = Path(args.log_path)/args.model
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)


    checkpoint_path = model_path/'checkpoints'

    if args.load_best_model:
        model_name = f'{args.model}_best.pth'
    else:
        model_name = f'{args.model}.pth'
    checkpoint_file = checkpoint_path/model_name

    config_file = model_path/f'{args.model}.yaml'
    config = mlconfig.load(str(config_file))
    # Must set to False, since we want to evaluate on test instead of val set
    config.dataset.valset = False
    config.dataset.eval_batch_size = 100

    # init logger
    if args.attack_choice == 'AA' and args.aa_type != 'Standard':
        attack_name = f'AA-{args.aa_type}'
    else:
        attack_name = args.attack_choice
            
    log_file_path = log_path / \
        f'{args.model}_eval_{args.norm}_{attack_name}.log'
    logger = util.setup_logger(name=args.model, 
                               log_file=str(log_file_path),
                               console=not args.progress_bar)

    # set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        device_list = [torch.cuda.get_device_name(
            i) for i in range(0, torch.cuda.device_count())]
        logger.info(f"GPU List: {device_list}")
    else:
        device = torch.device('cpu')
    args.device = device

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    start = time.time()
    main()
    end = time.time()
    eval_time = (end - start) / 86400
    payload = f"Running Cost {eval_time:4f} Days"
    logger.info(payload)
