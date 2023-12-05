from typing import Any, Callable

from loguru import logger
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import models


class LossDeepCoral:
    def __init__(
        self,
        criterion=None,
        lambda_classify=1,
        lambda_coral=0.5,
        *,
        lambda_coral_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Parameters
        ---
        criterion : (Tensor, List[int]) -> Tensor, default = torch.nn.CrossEntropyLoss().cuda()
            loss function
        lambda_coral_func : Callable[[torch.Tensor], torch.Tensor] | None, default = None
            coral_loss = lambda_coral_func(coral_loss), default is not change.
            sometimes, lambda_coral_func = lambda x:x**(1/7) maybe useful.
        """
        self.__criterion = criterion or torch.nn.CrossEntropyLoss().cuda()
        self.lambda_classify = lambda_classify
        self.lambda_coral = lambda_coral
        self.func_coral = lambda_coral_func

    def __call__(
        self,
        output: torch.Tensor,
        labels: list[int],
    ) -> Any:
        batch_size: int = output.data.shape[0]
        half_batch_size: int = batch_size // 2

        source_output = output[:half_batch_size]
        target_output = output[half_batch_size:]
        source_labels = labels[:half_batch_size]

        classification_loss = self.__criterion(source_output, source_labels)
        # classification_loss = self.__criterion(output, target)
        coral_loss = self._coral(source_output, target_output)
        if self.func_coral is not None:
            coral_loss = self.func_coral(coral_loss)
        logger.debug(
            f"DEEP_CORAL_COUNTING--{float(self.lambda_classify)=}--{float(classification_loss)=}--{float(self.lambda_coral)=}--{float(coral_loss)=}"
        )
        return (
            self.lambda_classify * classification_loss + self.lambda_coral * coral_loss
        )

    @property
    def classification_criterion(self):
        return self.__criterion

    @staticmethod
    def _coral(
        source: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        output of model is some feature, and feature also is probability of a image in a class.
        for the feature-array covariance
            1. zero meanlize, using mean(feature-array)
            2. sum(feature-array) in batch
        which means the covariance of each class-pair

        loss is the diffence of 2 covariance-matrix,
            covariance-matrix can represent the shape of class_num-dimonsion array(feature)
            the loss become lower, the shape be come similar
        """
        # source.shape = (batch_size, class_num)
        class_num = source.data.shape[1]

        # source covariance
        source_mean = torch.mean(source, 0, keepdim=True)  # shape = (1, class_num)
        source_zero_mean = source_mean - source  # shape = (batch_size, class_num)
        # covariance for each output
        source_covariance = (
            source_zero_mean.t() @ source_zero_mean
        )  # shape = (class_num, class_num) = (class_num, batch_size) @ (batch_size, class_num)

        # target covariance
        target_mean = torch.mean(target, 0, keepdim=True)  # shape = (1, class_num)
        target_zero_mean = target_mean - target  # shape = (batch_size, class_num)
        # covariance for each output
        target_covariance = (
            target_zero_mean.t() @ target_zero_mean
        )  # shape = (class_num, class_num) = (class_num, batch_size) @ (batch_size, class_num)

        # frobenius norm-L2 of distance between source and target
        loss = torch.norm(source_covariance - target_covariance)  # shape = (), scalar
        loss = loss / (4 * class_num * class_num)

        return loss

    @classmethod
    def coral(cls, output: torch.Tensor, *args, **kwargs):
        """
        Using Combined Output
        """
        batch_size: int = output.data.shape[0]
        half_batch_size: int = batch_size // 2

        source_output = output[:half_batch_size]
        target_output = output[half_batch_size:]
        return cls._coral(source_output, target_output)


class LmmdLoss:
    def __init__(
        self,
        num_classes: int,
        lambda_classify=1,
        lambda_lmmd=1,
        mu: float = 2.0,
        sigma: float | None = None,
        kernel_num: int = 5,
        criterion=None,
        *,
        lambda_lmmd_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        r"""
        Parameters
        ---
        num_classes : int
            number of classes
        mu : float, default = 2.0
            \mu of guassian distribution (also control \sigma)
        sigma : float, default = None
            if didn't set the sigma, then automatic calculate it
        kernel_num : int, default = 5
            number of kernels
        criterion : (Tensor, List[int]) -> Tensor, default = torch.nn.CrossEntropyLoss().cuda()
            loss function
        lambda_lmmd_func : Callable[[torch.Tensor], torch.Tensor] | None, default = None
            lmmd_loss = lambda_lmmd_func(lmmd_loss), default is not change.
            sometimes, lambda_lmmd_func = lambda x:x**(1/7) maybe useful.
        """
        self.lambda_classify = lambda_classify
        self.lambda_lmmd = lambda_lmmd
        self.__lambda_lmmd_func = lambda_lmmd_func
        self.__num_classes = num_classes
        self.__mu = mu
        self.__sigma = sigma
        self.__kernel_num = kernel_num
        self.__criterion = criterion or torch.nn.CrossEntropyLoss().cuda()

    def __call__(
        self,
        output: torch.Tensor,
        labels: list[int],
    ) -> Any:
        batch_size: int = output.data.shape[0]
        half_batch_size: int = batch_size // 2

        source_output = output[:half_batch_size]
        target_output = output[half_batch_size:]
        source_labels = labels[:half_batch_size]

        target_output = F.softmax(target_output, dim=1)

        classification_loss = self.__criterion(source_output, source_labels)
        lmmd_loss = self._cal_mmd_loss(source_output, target_output, source_labels)
        if self.__lambda_lmmd_func is not None:
            lmmd_loss = self.__lambda_lmmd_func(lmmd_loss)
        logger.debug(
            f"DEEP_LMMD_COUNTING--{float(self.lambda_classify)=}--{float(classification_loss)=}--{float(self.lambda_lmmd)=}--{float(lmmd_loss)=}"
        )
        return self.lambda_classify * classification_loss + self.lambda_lmmd * lmmd_loss

    def _cal_mmd_loss(
        self, sourse: torch.Tensor, target: torch.Tensor, source_label: list[int]
    ) -> torch.Tensor:
        r"""
        Parameters
        ---
        source : torch.Tensor
            source output, shape = (batch_size, class_num)
        target : torch.Tensor
            target output, shape = (batch_size, class_num)

        Returns
        ---
        loss : torch.Tensor
            shape = (), is a scalar
        """
        batch_size: int = len(source_label)
        device = target.device

        w_ss, w_tt, w_st = self.get_weight(source_label, target, self.__num_classes)

        kernels = self.guassian_kernel(
            sourse,
            target,
            mu=self.__mu,
            sigma=self.__sigma,
            kernel_num=self.__kernel_num,
        )

        loss: torch.Tensor = torch.Tensor([0]).to(device)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss

        k_ss = kernels[:batch_size, :batch_size]
        k_tt = kernels[batch_size:, batch_size:]
        k_st = kernels[:batch_size, batch_size:]

        loss += torch.sum(w_ss * k_ss + w_tt * k_tt - 2 * w_st * k_st)
        return loss

    @staticmethod
    def guassian_kernel(
        sourse: torch.Tensor,
        target: torch.Tensor,
        mu: float = 2.0,
        sigma: float | None = None,
        kernel_num: int = 5,
    ) -> torch.Tensor:
        r"""
        Parameters
        ---
        source : torch.Tensor
            source, shape = (batch_size, class_num)
        target : torch.Tensor
            target, shape = (batch_size, class_num)
        mu : float
            \mu of guassian distribution (also control \sigma)
        sigma : float
            if didn't set the sigma, then automatic calculate it
        kernel_num : int
            number of kernels

        Returns
        ---
        kernel : torch.Tensor
            shape = (batch_size_sum, batch_size_sum)
        """
        device = sourse.device

        batch_size_sum = sourse.shape[0] + target.shape[0]
        combined = torch.cat([sourse, target], dim=0)

        # expand shape to (batch_size, batch_size, class_num)
        combined_expand0 = combined.unsqueeze(0).expand(
            *(combined.shape[i] for i in (0, 0, 1))
        )  # expand(repeat) at dim=0
        combined_expand1 = combined.unsqueeze(1).expand(
            *(combined.shape[i] for i in (0, 0, 1))
        )  # expand(repeat) at dim=1

        # norm_l2 = ((combined_expand0 - combined_expand1)**2).sum(2)
        norm_l2 = torch.norm(combined_expand0 - combined_expand1, dim=2)

        # adjust the sigma of kernel
        if sigma is not None:
            bandwidth = sigma
        else:
            bandwidth = torch.sum(norm_l2.data) / (batch_size_sum**2 - batch_size_sum)

        # set sigma = mean, bandwidth_arr = bandwidth * (kernel_num**x)
        # >>> sigma = 1, mu = 2, kernel_num = 5
        # <<< bandwidth_arr = [0.25, 0.5, 1, 2, 4]
        x = torch.arange(kernel_num).to(device)
        bandwidth /= mu ** (kernel_num // 2)
        bandwidth_arr = bandwidth * (kernel_num**x)

        # guassian_kernel
        # shape: (kernel_num) -> (batch_size_sum, batch_size_sum, kernel_num)
        x = bandwidth_arr.expand(
            norm_l2.shape[0], norm_l2.shape[0], bandwidth_arr.shape[0]
        )
        # shape: (batch_size_sum, batch_size_sum) -> (batch_size_sum, batch_size_sum, kernel_num)
        x2 = norm_l2.unsqueeze(2).expand(
            norm_l2.shape[0], norm_l2.shape[0], bandwidth_arr.shape[0]
        )
        kernel = torch.exp(-x2 / x)

        return torch.sum(kernel, dim=2)  # shape = (batch_size_sum, batch_size_sum)

    @staticmethod
    def get_weight(
        s_label: list[int], t_pred: torch.Tensor, num_classes: int = 65
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ---
        s_label : list[int]
            list of labels, label is some `int` number, and not bigger than num_classes
        t_pred : torch.Tensor
            model output
        num_classes : int
            numbers of classes

        Returns
        ---
        weight_ss, weight_tt, weight_st
            part of weight matrix, weight_st == weight_ts
        """
        batch_size = len(s_label)
        device = t_pred.device

        # s_label_num: torch.Tensor = torch.Tensor(s_label)  # shape = (batch_size)
        s_label_hot: torch.Tensor = F.one_hot(s_label, num_classes).to(
            device
        )  # shape = (batch_size, class_num)
        s_projected_hot: torch.Tensor = torch.reshape(
            torch.sum(s_label_hot, axis=0), (-1,)
        )  # shape = (class_num)
        # maximum = 1 for avoid devided by 0
        s_label_hot_normlized: torch.Tensor = s_label_hot / torch.maximum(
            s_projected_hot, torch.Tensor([1]).to(device)
        )  # shape = (batch_size, class_num)

        # t_label_num: torch.Tensor = torch.argmax(
        #     t_pred.data, dim=1
        # )  # shape = (batch_size)
        t_label_hot: torch.Tensor = t_pred.data  # shape = (batch_size, class_num)
        t_projected_hot: torch.Tensor = torch.reshape(
            torch.sum(t_label_hot, axis=0), (-1,)
        )  # shape = (class_num)
        # maximum = 1 for avoid devided by 0
        t_label_hot_normlized: torch.Tensor = t_label_hot / torch.maximum(
            t_projected_hot, torch.Tensor([1]).to(device)
        )  # shape = (batch_size, class_num)

        st_both_hot = (s_projected_hot != 0) & (
            t_projected_hot != 0
        )  # shape = (class_num), dtype = bool
        mask_arr = st_both_hot.expand(
            (batch_size, num_classes)
        )  # shape: (class_num)->(batch_size, num_classes)
        t_label_hot_normlized_masked = (
            t_label_hot_normlized * mask_arr
        )  # shape = (batch_size, num_classes)
        s_label_hot_normlized_masked = (
            s_label_hot_normlized * mask_arr
        )  # shape = (batch_size, num_classes)

        # weight_st == weight_ts
        weight_ss = s_label_hot_normlized_masked @ s_label_hot_normlized_masked.T
        weight_tt = t_label_hot_normlized_masked @ t_label_hot_normlized_masked.T
        weight_st = s_label_hot_normlized_masked @ t_label_hot_normlized_masked.T

        both_hot_num = torch.sum(st_both_hot)
        if both_hot_num != 0:
            # maximum = 1 for avoid devided by 0
            # normalize
            weight_ss: torch.Tensor = weight_ss / torch.maximum(
                both_hot_num, torch.Tensor([1]).to(device)
            )
            weight_tt: torch.Tensor = weight_tt / torch.maximum(
                both_hot_num, torch.Tensor([1]).to(device)
            )
            weight_st: torch.Tensor = weight_st / torch.maximum(
                both_hot_num, torch.Tensor([1]).to(device)
            )
        else:
            weight_ss: torch.Tensor = torch.Tensor([0]).to(device)
            weight_tt: torch.Tensor = torch.Tensor([0]).to(device)
            weight_st: torch.Tensor = torch.Tensor([0]).to(device)

        return (
            weight_ss,
            weight_tt,
            weight_st,
        )
