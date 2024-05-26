import mlconfig
import argparse
import util
import time
import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import tqdm
import pickle
import gc

import torch.nn.functional as F
from torchvision import datasets, transforms

from dataset import CIFAR10C, DatasetGenerator


# used to register the models
import models
mlconfig.register(DatasetGenerator)


def setup_args():
    parser = argparse.ArgumentParser(
        description="Evaluate network on cifar-10-c"
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default="arch_001")
    parser.add_argument('--model-path', type=str, default='models_home')
    parser.add_argument('--log-path', type=str)
    parser.add_argument('--load-best-model',
                        action='store_true', default=False)
    parser.add_argument('--progress-bar', action='store_true')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()
    return args


def test(model, test_loader, progress_bar=False):
    """Evaluate network on given dataset."""
    metrics = defaultdict(util.AverageMeter)
    model.eval()

    if progress_bar:
        _data_loader = tqdm.tqdm(test_loader)
    else:
        _data_loader = test_loader

    with torch.no_grad():
        for (images, labels) in _data_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            bs = images.size(0)
            logits = model(images)
            loss = F.cross_entropy(logits, labels).item()
            pred = logits.max(1)[1].detach()
            acc = (pred == labels).float().mean().item()

            metrics['test_loss'].update(loss, n=bs)
            metrics['test_acc'].update(acc, n=bs)

    return metrics


def test_c(model, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_metrics = defaultdict(util.AverageMeter)

    test_transform = transforms.ToTensor()

    for cname in CIFAR10C.CORRUPTIONS:
        test_data = CIFAR10C(
            root=base_path,
            corruption=cname,
            transform=test_transform,
        )
        for level in range(1, 6):
            test_data.set_level(level)
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True)

            metrics = test(model, test_loader, args.progress_bar)
            logger.info(
                f"{cname} (level={level}): Loss: {metrics['test_loss'].avg:.4f}, Acc: {metrics['test_acc'].percent:.2f}")

            corruption_metrics[f'{cname}_{level}'] = metrics

        gc.collect()
        torch.cuda.empty_cache()

    avg_acc = np.mean(
        [metrics['test_acc'].percent for metrics in corruption_metrics.values()])
    logger.info(f"Average Accuracy: {avg_acc:.4f}")

    return corruption_metrics


def main():
    model = config.model().to(args.device)
    # logger.info(model)

    # use cifar10c dataset
    checkpoint = util.load_model(
        filename=checkpoint_file, model=model, optimizer=None, alpha_optimizer=None, scheduler=None)

    model.eval()
    model = torch.nn.DataParallel(model).to(args.device)

    corruption_metrics = test_c(model, base_path="datasets/cifar10c")

    with (log_path/f"cifar10c.pkl").open('wb') as f:
        pickle.dump(corruption_metrics, f)


if __name__ == '__main__':
    args = setup_args()
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

    log_file_path = log_path/f'{args.model}_eval_cifar10c.log'
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

    config.set_immutable()

    start = time.time()
    main()
    end = time.time()
    eval_time = (end - start) / 3600
    payload = f"Running Cost {eval_time:.2f} Hours"
    logger.info(payload)
