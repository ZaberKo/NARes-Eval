import mlconfig
import argparse
import util
import dataset
import time
import os
import torch
import numpy as np
from base_attackers import Evaluator
from torch.autograd import Variable
from torchprofile import profile_macs
from auto_attack.autoattack import AutoAttack
mlconfig.register(dataset.DatasetGenerator)

# for instance python ./eval_robustness.py --config-path /research/hal-huangs88/codes/Robust_NASBench/ablation_dir/nasbench/small/r1 \
# --load-best-model  --version arch_001 --seed 0

parser = argparse.ArgumentParser(description='Adversarial Attack Evaluate: Linf (L2)-version of FGSM, PGD20, CW40, AutoAttack with epsilon@8/255; L2 ')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--version', type=str, default="DARTS_Search")
parser.add_argument('--config-path', type=str, default='configs')
parser.add_argument('--attack-choice', type=str, default='Base', choices=['Base', "AA"], help="Base: FGSM, PGD, CW; Or, AA")
parser.add_argument('--norm', type=str, default='Linf', choices=["L2", "Linf"])
parser.add_argument('--load-best-model', action='store_true', default=False)
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size, for pgd etc., step size is 10x smaller than epsilon')
# Autoattack augments 
parser.add_argument('--aa-type', type=str, default='Standard', choices=["Compact", "Standard"], 
                    help='Compact: only includes two attacker which is almost closed to the Standard version in most case, and it is cheaper; Standard: Common used')

args = parser.parse_args()

if args.epsilon > 1:
    args.epsilon = args.epsilon / 255
    args.step_size = args.step_size / 255

# get the corresponding directory 
checkpoint_path = "{}/checkpoints".format(args.config_path)
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
config_file = os.path.join(args.config_path, args.version) + '.yaml'
config = mlconfig.load(config_file)
config['dataset']['valset'] = False     # Must set to False, since we want to evaluate on test instead of val set
config['dataset']['eval_batch_size'] = 100

# init logger
log_file_path = os.path.join(args.config_path, args.version)
logger = util.setup_logger(name=args.version, log_file=log_file_path + '_eval_{}-{}.log'.format(args.norm, args.attack_choice))

# set the seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')
args.device= device


def l2_eval(data_loader, model, evaluator, log=True):
    natural_count, fgsm_count, pgd_count, cw_count, total, stable_count = 0, 0, 0, 0, 0, 0
    loss_meters = util.AverageMeter()
    lip_meters = util.AverageMeter()
    model.eval()
    for i, (images, labels) in enumerate(data_loader["test_dataset"]):
        images, labels = images.to(device), labels.to(device)
        images, labels = Variable(images, requires_grad=True), Variable(labels)
        total += images.size(0)

        # L2-PGD attacker with 20 steps
        pgd = evaluator._l2pgd_whitebox(images, labels, num_steps=20, step_size=args.step_size)
        acc, acc_pgd, loss, stable, X_pgd = pgd
        natural_count += acc
        pgd_count += acc_pgd
        stable_count += stable
        local_lip = util.local_lip(model, images, X_pgd).item()
        lip_meters.update(local_lip)
        loss_meters.update(loss)

        # L2-CW attacker with 40 steps
        cw = evaluator._l2pgd_cw_whitebox(images, labels, num_steps=40, step_size=args.step_size)
        _, acc_cw, _, _, _ = cw
        cw_count += acc_cw

        if log:
            payload = 'LIP: %.4f\tStable Count: %d\tNatural Acc: %.2f\tPGD Acc: %.2f\tCW Acc: %.2f' % (
                local_lip, stable_count,  (natural_count / total) * 100, (pgd_count / total) * 100, (cw_count / total) * 100)
            logger.info(payload)

    natural_acc = (natural_count / total) * 100
    fgsm_acc = 0
    pgd_acc = (pgd_count / total) * 100
    cw_acc = (cw_count / total) * 100
    stable_acc = (stable_count / total) * 100 
    payload = 'Natural Correct Count: %d/%d, Acc: %.2f ' % (natural_count, total, natural_acc)
    logger.info(payload)
    payload = 'PGD with 20 steps Correct Count: %d/%d, Acc: %.2f ' % (pgd_count, total, pgd_acc)
    logger.info(payload)
    payload = 'CW with 40 steps Correct Count: %d/%d, Acc: %.2f ' % (pgd_count, total, cw_acc)
    logger.info(payload)
    payload = 'PGD with 20 steps Loss Avg: %.2f; LIP Avg: %.4f; Stable Acc: %.2f' % (loss_meters.avg, lip_meters.avg, stable_acc)
    logger.info(payload)
    return natural_acc, fgsm_acc, pgd_acc, cw_acc, stable_acc, lip_meters.avg


def linf_eval(data_loader, model, evaluator, log=True):
    natural_count, fgsm_count, pgd_count, cw_count, total, stable_count = 0, 0, 0, 0, 0, 0
    loss_meters = util.AverageMeter()
    lip_meters = util.AverageMeter()
    model.eval()
    for i, (images, labels) in enumerate(data_loader["test_dataset"]):
        images, labels = images.to(device), labels.to(device)
        images, labels = Variable(images, requires_grad=True), Variable(labels)
        total += images.size(0)

        # Linf-FGSM with 1 step. For FGSM, where only one step, usually, step size = epsilon
        fgsm = evaluator._fgsm_whitebox(images, labels, num_steps=1, step_size=args.epsilon)
        _, acc_fgsm, _, _, _ = fgsm
        fgsm_count += acc_fgsm

        # Linf-PGD attacker with 20 steps
        pgd = evaluator._pgd_whitebox(images, labels, num_steps=20, step_size=args.step_size)
        acc, acc_pgd, loss, stable, X_pgd = pgd
        natural_count += acc
        pgd_count += acc_pgd
        stable_count += stable
        local_lip = util.local_lip(model, images, X_pgd).item()
        lip_meters.update(local_lip)
        loss_meters.update(loss)

        # Linf-CW attacker with 40 steps
        cw = evaluator._pgd_whitebox(model, images, labels, num_steps=40, step_size=args.step_size)
        _, acc_cw, _, _, _ = cw
        cw_count += acc_cw

        if log:
            payload = 'LIP: %.4f\tStable Count: %d\tNatural Acc: %.2f\tFGSM Acc: %.2f\tPGD Acc: %.2f\tCW Acc: %.2f' % (
                local_lip, stable_count,  (natural_count / total) * 100, (fgsm_count / total) * 100, (pgd_count / total) * 100, (cw_count / total) * 100)
            logger.info(payload)

    natural_acc = (natural_count / total) * 100
    fgsm_acc = (fgsm_count / total) * 100
    pgd_acc = (pgd_count / total) * 100
    cw_acc = (cw_count / total) * 100
    stable_acc = (stable_count / total) * 100 
    payload = 'Natural Correct Count: %d/%d, Acc: %.2f ' % (natural_count, total, natural_acc)
    logger.info(payload)
    payload = 'FGSM with 1 step Correct Count: %d/%d, Acc: %.2f ' % (fgsm_count, total, fgsm_acc)
    logger.info(payload)
    payload = 'PGD with 20 steps Correct Count: %d/%d, Acc: %.2f ' % (pgd_count, total, pgd_acc)
    logger.info(payload)
    payload = 'CW with 40 steps Correct Count: %d/%d, Acc: %.2f ' % (pgd_count, total, cw_acc)
    logger.info(payload)
    payload = 'PGD with 20 steps Loss Avg: %.2f; LIP Avg: %.4f; Stable Acc: %.2f' % (loss_meters.avg, lip_meters.avg, stable_acc)
    logger.info(payload)
    return natural_acc, fgsm_acc, pgd_acc, cw_acc, stable_acc, lip_meters.avg


def main():
    # Load Search Version Genotype
    model = config.model().to(args.device)
    logger.info(model)

    # Setup ENV
    data_loader = config.dataset().getDataLoader()
    profile_inputs = (torch.randn([1, 3, 32, 32]).to(args.device),)
    flops = profile_macs(model, profile_inputs) / 1e6
    params = util.count_parameters_in_MB(model)

    config.set_immutable()
    for key in config:
        logger.info("%s: %s" % (key, config[key]))
    logger.info("param size = %fMB", params)
    logger.info("flops: %.4fM" % flops)
    logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

    # load trained weights
    filename = checkpoint_path_file + '_best.pth' if args.load_best_model else checkpoint_path_file + '.pth'
    checkpoint = util.load_model(filename=filename, model=model, optimizer=None, alpha_optimizer=None, scheduler=None)
    ENV = checkpoint['ENV']
    ENV['params': params, 'flops': flops]
    logger.info("File %s loaded!" % (filename))

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model = torch.nn.DataParallel(model).to(args.device)

    # Base attackers: FGSM, PGD, CW
    if args.attack_choice == 'Base':
        evaluator = Evaluator(model, data_loader, logger, config, random_start=True, epsilon=args.epsilon, device=args.device)
        if args.norm == 'Linf':
            natural_acc, fgsm_acc, pgd_acc, cw_acc, stable_acc, lip = linf_eval(data_loader, model, evaluator)
        else:
            natural_acc, fgsm_acc, pgd_acc, cw_acc, stable_acc, lip = l2_eval(data_loader, model, evaluator)

        ENV['{}_natural_acc'.format(args.norm)] = natural_acc
        ENV['{}_fgsm_acc'.format(args.norm)] = fgsm_acc
        ENV['{}_pgd20_acc'.format(args.norm)] = pgd_acc
        ENV['{}_cw40_acc'.format(args.norm)] = cw_acc
        ENV['{}_pgd20_stable'.format(args.norm)] = stable_acc
        ENV['{}_pgd20_lip'.format(args.norm)] = lip

    else:
        # reshape dataloader for adapted to AA
        x_test = [x for (x, y) in data_loader['test_dataset']]
        x_test = torch.cat(x_test, dim=0)
        y_test = [y for (x, y) in data_loader['test_dataset']]
        y_test = torch.cat(y_test, dim=0)

        adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, logger=logger, verbose=True, device=args.device)
        logger.info('=' * 20 + 'AA with {} Attack Eval'.format(args.norm) + '=' * 20)
        if args.aa_type == 'Compact':       # only evaluate two attacker instead of four when use standard (default)
            adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
        print("  ### Evaluate {}-AA attackers ###  ".format(adversary.attacks_to_run))
        adv_imgs, robust_accuracy = adversary.run_standard_evaluation(x_test, y_test, bs=config.dataset.eval_batch_size)

        robust_accuracy = robust_accuracy * 100
        logger.info('%s-AA Accuracy: %.2f' % (args.norm, robust_accuracy))
        ENV['{}_aa_attack'.format(args.norm)] = robust_accuracy

    # print evaluation results
    logger(ENV)

    return


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    logger.info(payload)
