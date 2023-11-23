import torch
import torch.nn as nn

from torchattacks.attack import Attack


def cw_loss(output, target, kappa=0, mask_magnitude=10000.0):
    """
    CW loss (Marging loss).
    """
    target_onehot = torch.zeros_like(output)

    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    real = (target_onehot * output).sum(1)
    other = ((1. - target_onehot) * output -
             target_onehot * mask_magnitude).max(1)[0]

    loss = - torch.clamp(real - other + kappa, min=0.)
    loss = torch.sum(loss)
    return loss


class PGD_CWLinf(Attack):
    def __init__(self, model, eps=8/255, kappa=0, steps=20, lr=0.003, random_start=True):
        super().__init__("CWLinf", model)

        self.kappa = kappa  # confidence
        self.eps = eps
        self.steps = steps
        self.lr = lr
        self.random_start = random_start
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            # if self.targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            #     cost = loss(outputs, labels)
            cost = cw_loss(outputs, labels, self.kappa)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.lr * grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class PGD_CWL2(Attack):
    def __init__(self, model, eps=8/255, kappa=0, steps=20, lr=0.003, random_start=True):
        super().__init__("CWLinf", model)

        self.kappa = kappa  # confidence
        self.eps = eps
        self.steps = steps
        self.lr = lr
        self.random_start = random_start
        self.supported_mode = ["default"]

        self.eps_for_division = 1e-10

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            # if self.targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            #     cost = loss(outputs, labels)
            cost = cw_loss(outputs, labels, self.kappa)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            grad_norms = (
                torch.norm(grad.view(batch_size, -1), p=2, dim=1)
                + self.eps_for_division
            )  # nopep8
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.lr * grad

            delta = adv_images - images
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


# from torch.autograd import Variable
# import torch.optim as optim

# class PGD_CWLinf_Old:
#     def __init__(self, model, epsilon=8/255, steps=40, lr=0.003, random_start=True):
#         self.model=model
#         self.epsilon = epsilon
#         self.steps = steps
#         self.lr=lr
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.random_start = random_start

#     def __call__(self, imgs, labels):
#         adv_imgs = Variable(imgs.data, requires_grad=True)

#         def CWLoss(output, target, confidence=0):
#             """
#             CW loss (Marging loss).
#             """
#             num_classes = output.shape[-1]
#             target = target.data
#             target_onehot = torch.zeros(target.size() + (num_classes,))
#             target_onehot = target_onehot.to(self.device)
#             target_onehot.scatter_(1, target.unsqueeze(1), 1.)
#             target_var = Variable(target_onehot, requires_grad=False)
#             real = (target_var * output).sum(1)
#             other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
#             loss = - torch.clamp(real - other + confidence, min=0.)
#             loss = torch.sum(loss)
#             return loss

#         if self.random_start:
#             random_noise = torch.FloatTensor(
#                 *adv_imgs.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
#             adv_imgs = Variable(adv_imgs.data + random_noise, requires_grad=True)

#         for _ in range(self.steps):
#             opt = optim.SGD([adv_imgs], lr=1e-3)
#             opt.zero_grad()
#             loss = CWLoss(self.model(adv_imgs), labels)
#             loss.backward()
#             eta = self.lr * adv_imgs.grad.data.sign()
#             adv_imgs = Variable(adv_imgs.data + eta, requires_grad=True)
#             eta = torch.clamp(adv_imgs.data - imgs.data, -
#                             self.epsilon, self.epsilon)
#             adv_imgs = Variable(imgs.data + eta, requires_grad=True)
#             adv_imgs = Variable(torch.clamp(adv_imgs, 0, 1.0), requires_grad=True)

#         adv_imgs = Variable(adv_imgs.data, requires_grad=False)
#         return adv_imgs
