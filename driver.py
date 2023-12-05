"""
This file may publish to IceTiki@github later, I'm not quite sure which repository I choice, may be tikilib.
And this file also used to do my CV homework in 2023/11.

Usage
---
>>> from (model_name) import RunHandler, RunParams, RunTracer
"""
import time as _time
import typing as _typing
import dataclasses as _dataclasses
import json as _json
import pathlib as _pathlib

from loguru import logger as _logger
import tqdm as _tqdm

import torch as _torch
import torch.utils.data as _tc_data
import torch.nn as _nn
import torch.optim as _opt
import torch.optim.lr_scheduler as _lr_sche
import torchvision.models as _tvmodels


@_dataclasses.dataclass
class RunParams:
    model: _nn.Module
    dataloader: _tc_data.DataLoader = None
    criterion: _nn.Module = None
    optimizer: _opt.Optimizer = None
    scheduler: _lr_sche.LRScheduler = None
    arch: str = None
    device: _torch.device = _torch.device("cuda")


class Meter:
    """
    In short, it is used to count average. And, it save all the values (and weights).
    """

    def __init__(self, name: str = "meter", fmt: str = ":f") -> None:
        """
        Used to save values, and output formated average.

        Parameters
        ---
        name : str, default = "meter"
            name of meter
        fmt : str, default = ":f"
            how to formating data
        """
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def __str__(self) -> str:
        fmtstr = "{name}={val%s}({avg%s})" % (self.fmt, self.fmt)
        return fmtstr.format(
            **{"name": self.name, "val": self.__value_new, "avg": self.average}
        )

    @property
    def average(self) -> float:
        return self.sum / sum(self.__weights)

    @property
    def sum(self) -> float:
        return sum((i * j for i, j in zip(self.__values, self.__weights)))

    @property
    def values(self) -> list[float]:
        return self.__values

    @property
    def weights(self) -> list[float]:
        return self.__weights

    @property
    def summary(self) -> str:
        fmtstr = "{name}={avg%s}" % self.fmt
        return fmtstr.format(
            **{"name": self.name, "val": self.__value_new, "avg": self.average}
        )

    def add(self, value: float, weight: float = 1):
        """
        save new value and weight

        Parameters
        ---
        value : float
            value
        weight : float, default = 1
            weight of value
        """
        self.__value_new = value
        self.__values.append(float(value))
        self.__weights.append(float(weight))

    def __iadd__(self, args: float) -> _typing.Self:
        assert len(args) in (0, 1)
        args = map(float, args)

        self.add(*args)
        return self

    @_typing.overload
    def __iadd__(self, value: float) -> _typing.Self:
        ...

    @_typing.overload
    def __iadd__(self, value_and_weights: tuple[float, float]) -> _typing.Self:
        ...

    def reset(self):
        """
        reset all the values and weights
        """
        self.__value_new = float
        self.__values: list[float] = []
        self.__weights: list[float] = []


class RunTracer:
    """
    Trace the model running status.
    - model running time
    - data loading time
    - loss
    - accuracy 1
    - accuracy 5
    """

    def __init__(
        self,
        num_batches: int,
        prefix: str = "",
    ) -> None:
        """
        Parameters
        ---
        num_batches : int
            number of batch, equal to len(DataLoader)
        prefix : str
            prefix, used to output to logger.
        """
        self._model_time = Meter("model", ":6.3f")  # time of model dealing the Tensor
        self._load_time = Meter("load", ":6.3f")  # time of dataloader load the data
        self._losses = Meter("loss", ":.4e")  # loss
        self._acc1 = Meter("acc1", ":6.2f")  # accuracy 1
        self._acc5 = Meter("acc5", ":6.2f")  # accuracy 5

        self.prefix = prefix
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)

    @property
    def meters(self) -> tuple[Meter]:
        """
        return all the meters in this instants.
        """
        return (self._model_time, self._load_time, self._losses, self._acc1, self._acc5)

    @property
    def data(self) -> dict[str, dict[str, list[float]]]:
        return {i.name: {"values": i.values, "weights": i.weights} for i in self.meters}

    @property
    def aver_acc1(self) -> float:
        return self._acc1.average

    @property
    def aver_loss(self) -> float:
        return self._losses.average

    def log(self, batch) -> None:
        """
        output status to logger.debug
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries.extend(map(str, self.meters))
        _logger.debug(" ".join(entries))

    def log_summary(self) -> None:
        """
        output summary to logger.debug
        """
        entries = [f"* {self.prefix}"]
        entries.extend(map(lambda x: x.summary, self.meters))
        _logger.debug(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """
        Examples
        ---
        >>> _get_batch_fmtstr(234)
        <<< "[{:3d}/234]"
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class EpochHandler:
    """
    Use to train or validate model in single epoch.
    """

    class _EmptyWith:
        def __init__(self) -> None:
            pass

        def __enter__(self, *args, **kwargs) -> None:
            pass

        def __exit__(self, *args, **kwargs) -> None:
            pass

    def __init__(
        self,
        epoch: int,
        train: bool,
        run_params: RunParams,
        name: str = "Run",
    ) -> None:
        """
        epoch : int
            Which epoch is now. Just for logger output.
        train : bool
            If train is True, then, use params for training. Else, just validating.
        run_params : RunParams
            Parameters for running.
        name : str, default = "Run"
            Name, just for logger output.
        """
        self.epoch: int = epoch
        self.train: bool = train
        self.run_params = run_params
        self.name: str = name

        self.tracer = RunTracer(
            len(self.run_params.dataloader),
            f"{'Train' if self.train else 'Valid'} {self.name}: ",
        )

    @staticmethod
    def accuracy(
        output: _torch.Tensor, target: _torch.Tensor, top_k: tuple[int] = (1,)
    ) -> list[_torch.Tensor]:
        """
        Computes the accuracy over the k top predictions for the specified values of k
        Parameters
        ---
        output : torch.Tensor
            model output
        target : torch.Tensor
            target from dataloader
        top_k : tuple[int], default = (1,)
            list of k, about the accuracy needed.

        Returns
        ---
        list[torch.Tensor]
            scalar float tensor
        """
        with _torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def run(self) -> RunTracer:
        """
        Run the epoch!
        """
        trace: RunTracer = self.tracer
        time_anchor = _time.time()

        dataloader: _tc_data.DataLoader = self.run_params.dataloader
        model: _nn.Module = self.run_params.model

        if self.train:
            # switch to train mode
            model.train()
        else:
            # switch to evaluate mode
            model.eval()

        with self._EmptyWith() if self.train else _torch.no_grad():
            bar = _tqdm.tqdm(dataloader, desc=f"{self.name}[{self.epoch}]")
            for loader_yield in bar:
                # measure data loading time
                trace._load_time.add(_time.time() - time_anchor)

                # batch_train
                self.__batch_run(loader_yield)

                # measure elapsed time
                trace._model_time.add(_time.time() - time_anchor)
                time_anchor = _time.time()

                bar.desc = f"{self.name}[{self.epoch}](acc={trace.aver_acc1:6.4f}|loss={trace.aver_loss:.4e})"

        trace.log_summary()
        return trace

    def __batch_run(
        self,
        batch_data: _typing.Any,
    ):
        """
        Run for a batch of data.

        Parameters
        ---
        batch_data : Any
            self.dataloader yielded
        """
        tracer = self.tracer
        images, target = batch_data

        model: _nn.Module = self.run_params.model
        criterion: _nn.Module = self.run_params.criterion
        optimizer: _opt.Optimizer = self.run_params.optimizer
        device: _torch.device = self.run_params.device

        # move data to the same device as model
        images: _torch.Tensor = images.to(device, non_blocking=True)
        target: _torch.Tensor = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss: _torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = self.accuracy(output, target, top_k=(1, 5))
        tracer._losses.add(loss.item(), images.size(0))
        tracer._acc1.add(acc1[0], images.size(0))
        tracer._acc5.add(acc5[0], images.size(0))

        if self.train:
            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class RunHandler:
    """
    Use to train or validate model.
    """

    def __init__(
        self,
        train: bool,
        run_params: RunParams,
        epoch_range: range = range(30),
        name: str = "",
    ) -> None:
        """
        Parameters
        ---
        train : bool

        """
        self.train: bool = train
        self.name: str = name
        self.__run_params: RunParams = run_params
        self.__epoch_range: range = epoch_range

        self.__epoch: int = None
        self.__last_tracer: RunTracer = None
        self.__tracers: list[RunTracer] = []
        self.callbacks: list[_typing.Callable[[_typing.Self], None]] = []

    @property
    def is_train_end(self) -> bool:
        """
        Returns
        ---
        bool
            If this epoch is last epoch, return True. Else, return False.
        """
        epoch = self.epoch
        step = self.__epoch_range.step
        stop = self.__epoch_range.stop
        return (stop - epoch) * (stop - (epoch + step)) <= 0

    @property
    def epoch_range(self) -> range:
        return self.__epoch_range

    @property
    def run_params(self) -> RunParams:
        return self.__run_params

    @property
    def epoch(self) -> int:
        return self.__epoch

    @property
    def tracer_data(self) -> list[dict[str, dict[str, list[float]]]]:
        return [i.data for i in self.__tracers]

    @property
    def tracers(self) -> list[RunTracer]:
        return self.__tracers

    @property
    def last_tracer(self) -> RunTracer:
        return self.__last_tracer

    @property
    def tracer_json(self) -> str:
        return _json.dumps(self.tracer_data, ensure_ascii=False)

    def __len__(self):
        return len(self.__epoch_range)

    def __iter__(self) -> _typing.Generator[RunTracer, None, None]:
        """
        Running for each epochs.

        Yields
        ---
        RunTracer
        """
        return self.run()

    def __callback(self):
        """run all the callback function"""
        for callback in self.callbacks:
            callback(self)

    def add_callback(self, func: _typing.Callable[[_typing.Self], None]):
        self.callbacks.append(func)

    def run(self) -> _typing.Generator[RunTracer, None, None]:
        """
        Running for each epochs.

        Yields
        ---
        RunTracer

        Examples
        ---
        >>> for tracer in RunHandler(...):
        >>>     ...
        """
        for epoch in self.__epoch_range:
            self.__epoch = epoch

            tracer: RunTracer = EpochHandler(
                epoch,
                self.train,
                run_params=self.__run_params,
                name=self.name,
            ).run()

            scheduler = self.run_params.scheduler
            if scheduler is not None and not self.train:
                scheduler.step()

            self.__tracers.append(tracer)
            self.__last_tracer = tracer
            self.__callback()
            yield tracer

    def run_all(self) -> tuple[RunTracer]:
        return tuple(i for i in self)

    def save(self, path: _pathlib.Path, comments: str = ""):
        path = _pathlib.Path(path)

        model = self.run_params.model
        optimizer = self.run_params.optimizer
        scheduler = self.run_params.scheduler
        arch = self.run_params.arch
        epoch = self.epoch

        data = {
            "epoch": epoch,
            "arch": arch,
            "state_dict": model.state_dict(),
            "comments": comments,
        }

        data["optimizer"] = optimizer.state_dict() if optimizer is not None else None
        data["scheduler"] = scheduler.state_dict() if scheduler is not None else None
        data["acc1"] = (
            self.last_tracer.aver_acc1 if self.last_tracer is not None else None
        )

        # saving model
        _torch.save(data, path)

    def load(self, path: _pathlib.Path, load_epoch=False):
        path = _pathlib.Path(path)
        data: dict = _torch.load(path)

        if load_epoch:
            epoch = data["epoch"]
            self.__epoch = epoch
            old_range = self.__epoch_range
            self.__epoch_range = range(
                old_range.start + epoch + old_range.step,
                old_range.stop + epoch + old_range.step,
                old_range.step,
            )

        if data["state_dict"] is not None:
            self.run_params.model.load_state_dict(data["state_dict"])

        if (
            data.get("optimizer", None) is not None
            and self.run_params.optimizer is not None
        ):
            self.run_params.optimizer.load_state_dict(data["optimizer"])

        if (
            data.get("scheduler", None) is not None
            and self.run_params.scheduler is not None
        ):
            self.run_params.scheduler.load_state_dict(data["optimizer"])

        return data.get("comments", "")
