import mlconfig
import argparse
import util
import dataset
import time
import os
import torch
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
    parser.add_argument('--attack-choice', type=str, default='Base',
                        choices=['Base', "AA"], help="Base: (FGSM) + PGD + CW; AA")
    parser.add_argument('--norm', type=str, default='Linf',
                        choices=["L2", "Linf"])
    parser.add_argument('--load-best-model',
                        action='store_true', default=False)
    # parser.add_argument('--epsilon', default=8,
    #                     type=float, help='perturbation')
    # parser.add_argument('--step_size', default=0.8, type=float,
    #                     help='perturb step size, for pgd etc., step size is 10x smaller than epsilon')
    # Autoattack augments
    parser.add_argument('--aa-type', type=str, default='Standard', choices=["Compact", "Standard"],
                        help='Compact: only includes two attacker which is almost closed to the Standard version in most case, and it is cheaper; Standard: Common used')
    parser.add_argument('--progress-bar', action='store_true')

    args = parser.parse_args()

    if args.norm == 'Linf':
        args.epsilon = 8/255
        args.step_size = 0.8/255
    else:
        args.epsilon = 0.5
        args.step_size = 0.8/255

    return args


def get_ratio(metric: util.AverageMeter):
    return f'{round(metric.sum)}/{round(metric.count)}'


def l2_eval(data_loader, model, evaluator, progress_bar=False):
    metrics = defaultdict(util.AverageMeter)
    model.eval()

    if progress_bar:
        _data_loader = tqdm.tqdm(data_loader["test_dataset"])
    else:
        _data_loader = data_loader["test_dataset"]

    for i, (images, labels) in enumerate(_data_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        bs = images.size(0)

        predict_clean, natural_acc = evaluator.clean_acc(images, labels)
        metrics['natural_acc'].update(natural_acc, n=bs)

        # L2-PGD attacker with 20 steps
        acc_pgd, stable_pgd, adv_imgs_pgd = evaluator.l2pgd_whitebox(
            images, labels, predict_clean)
        metrics['pgd_acc'].update(acc_pgd, n=bs)
        metrics['pgd_stable'].update(stable_pgd, n=bs)
        local_lip = util.local_lip(model, images, adv_imgs_pgd).item()
        metrics['pgd_lip'].update(local_lip, n=bs)

        # L2-CW attacker with 40 steps
        acc_cw, stable_cw, _ = evaluator.l2cw_whitebox(
            images, labels, predict_clean)
        metrics['cw_acc'].update(acc_cw, n=bs)
        metrics['cw_stable'].update(stable_cw, n=bs)

        payload = f'Nature Acc: {metrics["natural_acc"].percent:.2f} PGD Acc: {metrics["pgd_acc"].percent:.2f} PGD LIP: {local_lip:.4f} CW Acc: {metrics["cw_acc"].percent:.2f}'
        logger.info(payload)
        if progress_bar:
            _data_loader.set_description(payload)

    logger.info(
        f'Natural Correct Count: {get_ratio(metrics["natural_acc"])}, Acc: {metrics["natural_acc"].percent:.2f}'
    )
    logger.info(
        f'PGD with 20 steps Correct Count: {get_ratio(metrics["pgd_acc"])}, Acc: {metrics["pgd_acc"].percent:.2f}, Stable: {metrics["pgd_stable"].percent:.2f} LIP: {metrics["pgd_lip"].avg:.4f}'
    )
    logger.info(
        f'CW with 40 steps Correct Count: {get_ratio(metrics["cw_acc"])}, Acc: {metrics["cw_acc"].percent:.2f}, Stable: {metrics["natural_acc"].percent:.2f}'
    )
    return metrics


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

        predict_clean, natural_acc = evaluator.clean_acc(images, labels)
        metrics['natural_acc'].update(natural_acc, n=bs)

        # Linf-FGSM with 1 step. For FGSM, where only one step, usually, step size = epsilon
        acc_fgsm, stable_fgsm, _ = evaluator.fgsm_whitebox(
            images, labels, predict_clean)
        metrics['fgsm_acc'].update(acc_fgsm, n=bs)
        metrics['fgsm_stable'].update(stable_fgsm, n=bs)

        # Linf-PGD attacker with 20 steps
        acc_pgd, stable_pgd, adv_imgs_pgd = evaluator.pgd_whitebox(
            images, labels, predict_clean)
        metrics['pgd_acc'].update(acc_pgd, n=bs)
        metrics['pgd_stable'].update(stable_pgd, n=bs)
        local_lip = util.local_lip(model, images, adv_imgs_pgd).item()
        metrics['pgd_lip'].update(local_lip, n=bs)

        # Linf-CW attacker with 40 steps
        acc_cw, stable_cw, _ = evaluator.cw_whitebox(
            images, labels, predict_clean)
        metrics['cw_acc'].update(acc_cw, n=bs)
        metrics['cw_stable'].update(stable_cw, n=bs)

        payload = f'Nature Acc: {metrics["natural_acc"].percent:.2f} FGSM Acc: {metrics["fgsm_acc"].percent:.2f} PGD Acc: {metrics["pgd_acc"].percent:.2f} PGD LIP: {local_lip:.4f} CW Acc: {metrics["cw_acc"].percent:.2f}'
        logger.info(payload)
        if progress_bar:
            _data_loader.set_description(payload)

    logger.info(
        f'Natural Correct Count: {get_ratio(metrics["natural_acc"])}, Acc: {metrics["natural_acc"].percent:.2f}'
    )
    logger.info(
        f'FGSM with 1 step Correct Count: {get_ratio(metrics["fgsm_acc"])}, Acc: {metrics["fgsm_acc"].percent:.2f}, Stable: {metrics["fgsm_stable"].percent:.2f}'
    )
    logger.info(
        f'PGD with 20 steps Correct Count: {get_ratio(metrics["pgd_acc"])}, Acc: {metrics["pgd_acc"].percent:.2f}, Stable: {metrics["pgd_stable"].percent:.2f} LIP: {metrics["pgd_lip"].avg:.4f}'
    )
    logger.info(
        f'CW with 40 steps Correct Count: {get_ratio(metrics["cw_acc"])}, Acc: {metrics["cw_acc"].percent:.2f}, Stable: {metrics["natural_acc"].percent:.2f}'
    )

    return metrics


def main():
    # Load Search Version Genotype
    model = config.model().to(args.device)
    logger.info(model)

    # Setup train_statedict
    data_loader = config.dataset().getDataLoader()
    profile_inputs = (torch.randn([1, 3, 32, 32]).to(args.device),)
    flops = profile_macs(model, profile_inputs) / 1e6
    params = util.count_parameters_in_MB(model)

    config.set_immutable()
    for key in config:
        logger.info(f"{key}: {config[key]}")
    logger.info(f"param size = {params}MB")
    logger.info(f"flops: {flops:.4f}M")
    logger.info(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(
            i) for i in range(0, torch.cuda.device_count())]
        logger.info(f"GPU List: {device_list}")

    # load trained weights

    checkpoint = util.load_model(
        filename=checkpoint_file, model=model, optimizer=None, alpha_optimizer=None, scheduler=None)
    train_statedict = checkpoint['ENV']
    train_statedict.update({'flops': flops, 'params': params})
    logger.info(f"File {checkpoint_file} loaded!")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model = torch.nn.DataParallel(model).to(args.device)

    # Base attackers: FGSM, PGD, CW
    if args.attack_choice == 'Base':
        evaluator = Evaluator(model, args.epsilon, args.step_size,
                              pgd_steps=20, cw_steps=40, pgdl2_steps=20, cwl2_steps=40)

        if args.norm == 'Linf':
            result_dict = linf_eval(
                data_loader, model, evaluator, args.progress_bar)
            train_statedict[f'{args.norm}_fgsm_acc'] = result_dict['fgsm_acc'].percent
        else:
            result_dict = l2_eval(
                data_loader, model, evaluator, args.progress_bar)

        train_statedict[f'{args.norm}_natural_acc'] = result_dict['natural_acc'].percent
        train_statedict[f'{args.norm}_pgd20_acc'] = result_dict['pgd_acc'].percent
        train_statedict[f'{args.norm}_cw40_acc'] = result_dict['cw_acc'].percent
        train_statedict[f'{args.norm}_pgd20_stable'] = result_dict['pgd_stable_acc'].percent
        train_statedict[f'{args.norm}_pgd20_lip'] = result_dict['pgd_lip'].avg

    else:
        # reshape dataloader for adapted to AA
        x_test = [x for (x, y) in data_loader['test_dataset']]
        x_test = torch.cat(x_test, dim=0)
        y_test = [y for (x, y) in data_loader['test_dataset']]
        y_test = torch.cat(y_test, dim=0)

        adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon,
                               logger=logger, verbose=True, device=args.device)
        logger.info(
            '=' * 20 + 'AA with {} Attack Eval'.format(args.norm) + '=' * 20)
        # only evaluate two attacker instead of four when use standard (default)
        if args.aa_type == 'Compact':
            adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
        print(f"  ### Evaluate {adversary.attacks_to_run}-AA attackers ###  ")
        adv_imgs, robust_accuracy = adversary.run_standard_evaluation(
            x_test, y_test, bs=config.dataset.eval_batch_size)

        robust_accuracy = robust_accuracy * 100
        logger.info(f'{args.norm}-AA Acc: {robust_accuracy:.2f}')
        train_statedict[f'{args.norm}_aa_attack'] = robust_accuracy

    # print evaluation results
    logger.info(train_statedict)


if __name__ == '__main__':
    args = setup_args()
    # if args.epsilon > 1:
    #     args.epsilon = args.epsilon / 255
    #     args.step_size = args.step_size / 255

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
    payload = f"Running Cost {eval_time:.2f} Days"
    logger.info(payload)
