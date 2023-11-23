import torchattacks
from attacks import PGD_CWLinf, PGD_CWL2


class Evaluator():
    def __init__(self, model, eps=8/255, step_size=0.8/255, pgd_steps=20, cw_steps=40, pgdl2_steps=20, cwl2_steps=40):
        self.model = model
        self.eps = eps
        self.step_size = step_size

        # Linf attackers
        self._fgsm_attacker = torchattacks.FGSM(self.model, eps=self.eps)
        self._pgd_attacker = torchattacks.PGD(
            self.model, eps=self.eps, alpha=self.step_size, steps=pgd_steps)
        self._cw_attacker = PGD_CWLinf(
            self.model, eps=self.eps, kappa=0, lr=self.step_size, steps=cw_steps)

        # L2 attackers
        self._pgdl2_attacker = torchattacks.PGDL2(
            self.model, eps=self.eps, alpha=self.step_size, steps=pgdl2_steps)
        # self._cwl2_attacker = torchattacks.CW(
        #     self.model, c=1, kappa=0, lr=self.step_size, steps=cwl2_steps)
        self._cwl2_attacker = PGD_CWL2(
            self.model, eps=self.eps, kappa=0, lr=self.step_size, steps=cwl2_steps)


        self._predict_clean = None

        self.model.eval()

    @property
    def predict_clean(self):
        self._predict_clean = self.model(self.imgs).max(1)[1].detach()

        return self._predict_clean

    def _get_pred(self, imgs, labels, predict_clean=None):
        predict = self.model(imgs).max(1)[1].detach()
        acc = (predict == labels).float().mean().item()

        if predict_clean is not None:
            stable = (predict == predict_clean).float().mean().item()
        else:
            stable = None

        return acc, stable

    def clean_acc(self, imgs, labels):
        predict_clean = self.model(imgs).max(1)[1].detach()
        acc = (predict_clean == labels.data).float().mean().item()

        return predict_clean, acc

    def fgsm_whitebox(self, imgs, labels, predict_clean=None):
        adv_img = self._fgsm_attacker(imgs, labels)

        adv_acc, stable = self._get_pred(adv_img, labels, predict_clean)

        return adv_acc, stable, adv_img

    def pgd_whitebox(self, imgs, labels, predict_clean=None):
        adv_img = self._pgd_attacker(imgs, labels)

        adv_acc, stable = self._get_pred(adv_img, labels, predict_clean)

        return adv_acc, stable, adv_img

    def cw_whitebox(self, imgs, labels, predict_clean=None):
        adv_img = self._cw_attacker(imgs, labels)

        adv_acc, stable = self._get_pred(adv_img, labels, predict_clean)

        return adv_acc, stable, adv_img

    def l2pgd_whitebox(self, imgs, labels, predict_clean=None):
        adv_img = self._pgdl2_attacker(imgs, labels)

        adv_acc, stable = self._get_pred(adv_img, labels, predict_clean)

        return adv_acc, stable, adv_img

    def l2cw_whitebox(self, imgs, labels, predict_clean=None):
        adv_img = self._cwl2_attacker(imgs, labels)

        adv_acc, stable = self._get_pred(adv_img, labels, predict_clean)

        return adv_acc, stable, adv_img
