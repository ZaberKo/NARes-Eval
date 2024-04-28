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

    args.norm = 'Linf'
    args.epsilon = 8/255
    args.step_size = 0.8/255

    return args

def get_ratio(metric: util.AverageMeter):
    return f'{round(metric.sum)}/{round(metric.count)}'

def linf_eval(data_loader, model, evaluator, progress_bar=False):
    metrics = defaultdict(util.AverageMeter)
    model.eval()
    if progress_bar:
        _data_loader = tqdm.tqdm(data_loader["test_dataset"])
    else:
        _data_loader = data_loader["test_dataset"]

    for i, (images, labels) in enumerate(_data_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        bs = images.size(0)

        test_loss, predict_clean, natural_acc = evaluator.clean_acc(
            images, labels)
        metrics['test_loss'].update(test_loss, n=bs)
        metrics['natural_acc'].update(natural_acc, n=bs)

        # # Linf-FGSM with 1 step. For FGSM, where only one step, usually, step size = epsilon
        # acc_fgsm, stable_fgsm, _ = evaluator.fgsm_whitebox(
        #     images, labels, predict_clean)
        # metrics['fgsm_acc'].update(acc_fgsm, n=bs)
        # metrics['fgsm_stable'].update(stable_fgsm, n=bs)

        # # Linf-PGD attacker with 20 steps
        # acc_pgd, stable_pgd, adv_imgs_pgd = evaluator.pgd_whitebox(
        #     images, labels, predict_clean)
        # metrics['pgd_acc'].update(acc_pgd, n=bs)
        # metrics['pgd_stable'].update(stable_pgd, n=bs)
        # local_lip = util.local_lip(model, images, adv_imgs_pgd).item()
        # metrics['pgd_lip'].update(local_lip, n=bs)

        # Linf-CW attacker with 40 steps
        acc_cw, stable_cw, adv_imgs_cw = evaluator.cw_whitebox(
            images, labels, predict_clean)
        metrics['cw_acc'].update(acc_cw, n=bs)
        metrics['cw_stable'].update(stable_cw, n=bs)
        local_lip = util.local_lip(model, images, adv_imgs_cw).item()
        metrics['cw_lip'].update(local_lip, n=bs)

        payload = f'Nature Acc: {metrics["natural_acc"].percent:.2f} Test Loss: {metrics["test_loss"].avg:.4f} CW Acc: {metrics["cw_acc"].percent:.2f} CW Stable: {metrics["cw_stable"].percent:.2f} CW LIP: {metrics["cw_lip"].avg:.4f}'
        logger.info(payload)
        if progress_bar:
            _data_loader.set_description(payload)

    logger.info(
        f'Natural Correct Count: {get_ratio(metrics["natural_acc"])}, Acc: {metrics["natural_acc"].percent:.2f} Test Loss: {metrics["test_loss"].avg:.4f}'
    )
    # logger.info(
    #     f'FGSM with 1 step Correct Count: {get_ratio(metrics["fgsm_acc"])}, Acc: {metrics["fgsm_acc"].percent:.2f}, Stable: {metrics["fgsm_stable"].percent:.2f}'
    # )
    # logger.info(
    #     f'PGD with 20 steps Correct Count: {get_ratio(metrics["pgd_acc"])}, Acc: {metrics["pgd_acc"].percent:.2f}, Stable: {metrics["pgd_stable"].percent:.2f} LIP: {metrics["pgd_lip"].avg:.4f}'
    # )
    logger.info(
        f'CW with 40 steps Correct Count: {get_ratio(metrics["cw_acc"])}, Acc: {metrics["cw_acc"].percent:.2f}, Stable: {metrics["cw_stable"].percent:.2f} LIP: {metrics["cw_lip"].avg:.4f}'
    )

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

    evaluator = Evaluator(model, args.epsilon, args.step_size,
                        pgd_steps=20, cw_steps=40, pgdl2_steps=20, cwl2_steps=40)
    if args.norm == 'Linf':
        result_dict = linf_eval(
            data_loader, model, evaluator, args.progress_bar)

    # logger.info(result_dict)


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
    log_file_path = log_path / \
        f'{args.model}_eval_test-loss.log'
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
    eval_time = (end - start) / 3600
    payload = f"Running Cost {eval_time:4f} Hours"
    logger.info(payload)
