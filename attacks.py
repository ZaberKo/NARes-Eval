import torch
import torch.nn as nn

from torchattacks.attack import Attack


class CWLinf(Attack):
    def __init__(self, model, eps=8/255, kappa=0, steps=20, lr=0.003):
        super().__init__("CWLinf", model)

        self.kappa = kappa # confidence
        self.eps = eps
        self.steps = steps
        self.lr = lr
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        random_noise = torch.zeros_like(
            images).uniform_(-self.eps, self.eps)

        adv_images = images + random_noise

        for step in range(self.steps):
            adv_images = adv_images.detach()
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = self.CWLoss(outputs, labels)
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]
            eta = self.lr * grad.sign()
            adv_images = adv_images + eta
            eta = torch.clamp(adv_images - images,
                              -self.eps, self.eps)
            adv_images = torch.clamp(images + eta, 0, 1)

        return adv_images

    def CWLoss(self, output, target):
        """
        CW loss (Marging loss).
        """
        target = target.data
        target_onehot = torch.zeros_like(output)

        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = target_onehot.clone().detach()
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = - torch.clamp(real - other + self.kappa, min=0.)
        loss = torch.sum(loss)
        return loss
