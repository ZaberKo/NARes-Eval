import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def local_lip(model, x, xp, top_norm=1, btm_norm=float('inf'), reduction='mean'):
    model.eval()
    down = torch.flatten(x - xp, start_dim=1)
    with torch.no_grad():
        if top_norm == "kl":
            criterion_kl = nn.KLDivLoss(reduction='none')
            top = criterion_kl(F.log_softmax(model(xp), dim=1),
                               F.softmax(model(x), dim=1))
            ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=btm_norm)
        else:
            top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
            ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm)

    if reduction == 'mean':
        return torch.mean(ret)
    elif reduction == 'sum':
        return torch.sum(ret)
    else:
        raise ValueError("Not supported reduction")



def setup_logger(name, log_file=None, level=logging.INFO, console=True):
    """To setup as many loggers as you want"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = '\t' + key + '=' + value
        else:
            display += '\t' + str(key) + '=%.4f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1/batch_size))
    return res


def save_model(filename, epoch, model, optimizer, alpha_optimizer, scheduler, save_best=False, **kwargs):
    # Torch Save State Dict
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'alpha_optimizer_state_dict': alpha_optimizer.state_dict() if alpha_optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, filename)
    if save_best:
        # pre, ext = os.path.splitext(filename)
        # pre += '_best'
        # filename = pre + ext
        torch.save(state, filename.replace('last', 'best'))
    return


def load_model(filename, model, optimizer, alpha_optimizer,  scheduler, **kwargs):
    # Load Torch State Dict
    checkpoints = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoints['model_state_dict'], strict=False)
    if optimizer is not None and checkpoints['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    if alpha_optimizer is not None and checkpoints['alpha_optimizer_state_dict'] is not None:
        alpha_optimizer.load_state_dict(checkpoints['alpha_optimizer_state_dict'])
    if scheduler is not None and checkpoints['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    return checkpoints


def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)

    @property
    def percent(self):
        return self.avg * 100



