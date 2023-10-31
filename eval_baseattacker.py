import mlconfig
import argparse
import datetime
import util
import models
import dataset
import time
import os
import torch
import shutil
import numpy as np
from base_evaluator import Evaluator
from torch.autograd import Variable
from torchprofile import profile_macs

mlconfig.register(dataset.DatasetGenerator)

# for instance python ./eval_robustness.py --config-path /research/hal-huangs88/codes/Robust_NASBench/ablation_dir/nasbench/small/r1 \
# --load-best-model  --version arch_001 --seed 0

parser = argparse.ArgumentParser(description='Evaluate linf-version of FGSM, PGD20, CW40  with epsilon@8/255')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--version', type=str, default="DARTS_Search")
parser.add_argument('--config-path', type=str, default='configs')
parser.add_argument('--load-best-model', action='store_true', default=False)
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size')

args = parser.parse_args()
if args.epsilon > 1:
    args.epsilon = args.epsilon / 255
    args.step_size = args.step_size / 255

# get the corresponding directory 
checkpoint_path = "{}/checkpoints".format(args.config_path)
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
config_file = os.path.join(args.config_path, args.version, '{}.yaml'.format(args.version))
config = mlconfig.load(config_file)
config['dataset']['valset'] = False
config['dataset']['eval_batch_size'] = 100

log_file_path = os.path.join(args.config_path, args.version)
logger = util.setup_logger(name=args.version, log_file=log_file_path + '_eval@{}-{}steps'.format(args.attack_choice, args.num_steps) + ".log")

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


def whitebox_eval(data_loader, model, evaluator, log=True):
    natural_count, fgsm_count, pgd_count, cw_count, total, stable_count = 0, 0, 0, 0, 0, 0
    loss_meters = util.AverageMeter()
    lip_meters = util.AverageMeter()
    print("    ###    Using {} ####    ".format(args.attack_choice))
    model.eval()
    for i, (images, labels) in enumerate(data_loader["test_dataset"]):
        images, labels = images.to(device), labels.to(device)
        images, labels = Variable(images, requires_grad=True), Variable(labels)
        total += images.size(0)
        # FGSM with 1 step
        fgsm = evaluator._fgsm_whitebox(model, images, labels, random_start=True, 
                                        epsilon=args.epsilon, num_steps=1, step_size=args.step_size)
        _, acc_fgsm, _, _, _ = fgsm
        fgsm_count += acc_fgsm

        # PGD attacker with 20 steps
        pgd = evaluator._pgd_whitebox(model, images, labels, random_start=True, 
                                      epsilon=args.epsilon, num_steps=20, step_size=args.step_size)
        acc, acc_pgd, loss, stable, X_pgd = pgd
        natural_count += acc
        pgd_count += acc_pgd
        stable_count += stable
        local_lip = util.local_lip(model, images, X_pgd).item()
        lip_meters.update(local_lip)
        loss_meters.update(loss)

        # CW attacker with 40 steps
        cw = evaluator._pgd_whitebox(model, images, labels, random_start=True,
                                      epsilon=args.epsilon, num_steps=40, step_size=args.step_size)
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
    model = config.model().to(device)
    logger.info(model)

    # Setup ENV
    data_loader = config.dataset().getDataLoader()
    evaluator = Evaluator(data_loader, logger, config)
    profile_inputs = (torch.randn([1, 3, 32, 32]).to(device),)
    flops = profile_macs(model, profile_inputs) / 1e6

    config.set_immutable()
    for key in config:
        logger.info("%s: %s" % (key, config[key]))
    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    logger.info("flops: %.4fM" % flops)
    logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'flops': flops,
           'train_history': [],
           'natural_acc': 0,
           'fgsm_acc': 0,
           'pgd20_acc': 0.0,
           'cw40_acc': 0.0,
           'pgd20_stable': 0.0,
           'pgd20_lip': 0.0,
           'genotype_list': 0.0
           }

    # load trained weights
    filename = checkpoint_path_file + '_best.pth' if args.load_best_model else checkpoint_path_file + '.pth'
    checkpoint = util.load_model(filename=filename, model=model, optimizer=None, alpha_optimizer=None, scheduler=None)
    ENV = checkpoint['ENV']
    logger.info("File %s loaded!" % (filename))

    model = torch.nn.DataParallel(model).to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    natural_acc, fgsm_acc, pgd_acc, cw_acc, stable_acc, lip = whitebox_eval(data_loader, model, evaluator)
    ENV['natural_acc'] = natural_acc
    ENV["fgsm_acc"] = fgsm_acc
    ENV['pgd20_acc'] = pgd_acc
    ENV['cw40_acc'] = cw_acc
    ENV['pgd20_stable'] = stable_acc
    ENV['pgd20_lip'] = lip
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
