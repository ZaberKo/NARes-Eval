import models
import util
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class Evaluator():
    def __init__(self, model, data_loader, logger, config, random_start=False, epsilon=0.031, device='cpu', lr=1e-3):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.random_start = random_start
        self.lr = lr
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        return

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        return

    def eval(self, epoch):
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
            start = time.time()
            log_payload = self.eval_batch(images=images, labels=labels)
            end = time.time()
            time_used = end - start
        display = util.log_display(epoch=epoch, global_step=i, time_elapse=time_used, **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        return

    def eval_batch(self, images, labels):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            pred = self.model(images)
            loss = self.criterion(pred, labels)
            acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
        self.loss_meters.update(loss.item(), n=images.size(0))
        self.acc_meters.update(acc.item(), n=images.size(0))
        self.acc5_meters.update(acc5.item(), n=images.size(0))
        payload = {"val_acc": acc.item(),
                   "val_acc_avg": self.acc_meters.avg,
                   "val_acc5": acc5.item(),
                   "val_acc5_avg": self.acc5_meters.avg,
                   "val_loss": loss.item(),
                   "val_loss_avg": self.loss_meters.avg}
        return payload

    def _fgsm_whitebox(self, imgs, labels, num_steps=20, step_size=0.003):
        predict_clean = self.model(imgs).data.max(1)[1].detach()
        acc = (predict_clean == labels.data).float().sum()
        adv_imgs = Variable(imgs.data, requires_grad=True)   # pertur

        opt = optim.SGD([adv_imgs], lr=self.lr)
        opt.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(self.model(adv_imgs), labels)
        loss.backward()
        eta = self.epsilon * adv_imgs.grad.data.sign()
        adv_imgs = Variable(adv_imgs.data + eta, requires_grad=True)
        eta = torch.clamp(adv_imgs.data - imgs.data, -self.epsilon, self.epsilon)
        adv_imgs = Variable(imgs.data + eta, requires_grad=True)
        adv_imgs = Variable(torch.clamp(adv_imgs, 0, 1.0), requires_grad=True)

        adv_imgs = Variable(adv_imgs.data, requires_grad=False)
        predict_adv = self.model(adv_imgs).data.max(1)[1].detach()
        acc_adv = (predict_adv == labels.data).float().sum()
        stable = (predict_adv.data == predict_clean.data).float().sum()
        return acc.item(), acc_adv.item(), loss.item(), stable.item(), adv_imgs


    def _pgd_whitebox(self, imgs, labels, num_steps=20, step_size=0.003):
        predict_clean = self.model(imgs).data.max(1)[1].detach()
        acc = (predict_clean == labels.data).float().sum()
        adv_imgs = Variable(imgs.data, requires_grad=True)

        if self.random_start:
            random_noise = torch.FloatTensor(*adv_imgs.shape).uniform_(-self.epsilon, self.epsilon).to(device)
            adv_imgs = Variable(adv_imgs.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([adv_imgs], lr=1e-3)
            opt.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(self.model(adv_imgs), labels)
            loss.backward()
            eta = step_size * adv_imgs.grad.data.sign()
            adv_imgs = Variable(adv_imgs.data + eta, requires_grad=True)
            eta = torch.clamp(adv_imgs.data - imgs.data, -self.epsilon, self.epsilon)
            adv_imgs = Variable(imgs.data + eta, requires_grad=True)
            adv_imgs = Variable(torch.clamp(adv_imgs, 0, 1.0), requires_grad=True)

        adv_imgs = Variable(adv_imgs.data, requires_grad=False)
        predict_adv = self.model(adv_imgs).data.max(1)[1].detach()
        acc_adv = (predict_adv == labels.data).float().sum()
        stable = (predict_adv.data == predict_clean.data).float().sum()
        return acc.item(), acc_adv.item(), loss.item(), stable.item(), adv_imgs
    

    def _pgd_cw_whitebox(self, imgs, labels, num_steps=20, step_size=0.003):
        predict_clean = self.model(imgs).data.max(1)[1].detach()
        acc = (predict_clean == labels.data).float().sum()
        adv_imgs = Variable(imgs.data, requires_grad=True)

        def CWLoss(output, target, confidence=0):
            """
            CW loss (Marging loss).
            """
            num_classes = output.shape[-1]
            target = target.data
            target_onehot = torch.zeros(target.size() + (num_classes,))
            target_onehot = target_onehot.to(self.device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.)
            target_var = Variable(target_onehot, requires_grad=False)
            real = (target_var * output).sum(1)
            other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
            loss = - torch.clamp(real - other + confidence, min=0.)
            loss = torch.sum(loss)
            return loss

        if self.random_start:
            random_noise = torch.FloatTensor(*adv_imgs.shape).uniform_(-self.epsilon, self.epsilon).to(device)
            adv_imgs = Variable(adv_imgs.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([adv_imgs], lr=1e-3)
            opt.zero_grad()
            loss = CWLoss(self.model(adv_imgs), labels)
            loss.backward()
            eta = step_size * adv_imgs.grad.data.sign()
            adv_imgs = Variable(adv_imgs.data + eta, requires_grad=True)
            eta = torch.clamp(adv_imgs.data - imgs.data, -self.epsilon, self.epsilon)
            adv_imgs = Variable(imgs.data + eta, requires_grad=True)
            adv_imgs = Variable(torch.clamp(adv_imgs, 0, 1.0), requires_grad=True)

        adv_imgs = Variable(adv_imgs.data, requires_grad=False)
        predict_adv = self.model(adv_imgs).data.max(1)[1].detach()
        acc_adv = (predict_adv == labels.data).float().sum()
        stable = (predict_adv.data == predict_clean.data).float().sum()
        return acc.item(), acc_adv.item(), loss.item(), stable.item(), adv_imgs
    

       # come from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py
    def _l2pgd_whitebox(self, imgs, labels, num_steps=20, step_size=0.003, eps_for_division=0.0031):
        predict_clean = self.model(imgs).data.max(1)[1].detach()
        acc = (predict_clean == labels.data).float().sum()
        adv_imgs = imgs.clone().detach()
        loss_func = torch.nn.CrossEntropyLoss()
        batch_size = imgs.size()[0]

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_imgs).normal_()
            d_flat = delta.view(adv_imgs.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_imgs.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.epsilon
            adv_imgs = torch.clamp(adv_imgs + delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_imgs.requires_grad = True
            outputs = self.model(adv_imgs)
            loss = loss_func(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_imgs, retain_graph=False, create_graph=False)[0]
            grad_norms = (torch.norm(grad.view(batch_size, -1), p=2, dim=1) + eps_for_division) 
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_imgs = adv_imgs.detach() + step_size * grad
            delta = adv_imgs - imgs

            # L2 bounded perturbation
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()

        adv_imgs.requires_grad = False
        predict_adv = self.model(adv_imgs).data.max(1)[1].detach()
        acc_adv = (predict_adv == labels.data).float().sum()
        stable = (predict_adv.data == predict_clean.data).float().sum()
        return acc.item(), acc_adv.item(), loss.item(), stable.item(), adv_imgs
    

    # come from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/cw.py
    def _l2pgd_cw_whitebox(self, imgs, labels, num_steps=20, step_size=0.003, c=1, kappa=0):
        predict_clean = self.model(imgs).data.max(1)[1].detach()
        acc = (predict_clean == labels.data).float().sum()

        # define some functions
        def tanh_space(x):
            return 1 / 2 * (torch.tanh(x) + 1)

        def atanh(x):
            return 0.5 * torch.log((1 + x) / (1 - x))
        
        def inverse_tanh_space(x):
            # torch.atanh is only for torch >= 1.7.0
            # atanh is defined in the range -1 to 1
            return atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

        # f-function in the paper
        def ffunc(outputs, labels):
            one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

            # find the max logit other than the target class
            other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
            # get the target class's logit
            real = torch.max(one_hot_labels * outputs, dim=1)[0]

            return torch.clamp((real - other), min=-kappa)

        best_adv_imgs = imgs.clone().detach()
        best_L2 = 1e10 * torch.ones((len(imgs))).to(self.device)
        prev_cost = 1e10
        dim = len(imgs.shape)
        
        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        w = inverse_tanh_space(imgs).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=0.01)

        for step in range(num_steps):
            # Get adversarial images
            adv_imgs = tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_imgs), Flatten(imgs)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = self.model(adv_imgs)
            f_loss = ffunc(outputs, labels).sum()
            loss = L2_loss + c * f_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            # If the attack is not targeted we simply make these two values unequal
            condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_imgs = mask * adv_imgs.detach() + (1 - mask) * best_adv_imgs

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(num_steps // 10, 1) == 0:
                if loss.item() > prev_loss:
                    break 
                prev_loss = loss.item()

        best_adv_imgs.requires_grad = True
        predict_adv = self.model(best_adv_imgs).data.max(1)[1].detach()
        acc_adv = (predict_adv == labels.data).float().sum()
        stable = (predict_adv.data == predict_clean.data).float().sum()
        return acc.item(), acc_adv.item(), loss.item(), stable.item(), best_adv_imgs
