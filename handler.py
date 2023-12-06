import time as _time
import typing as _typing
import pathlib as _pathlib


from loguru import logger as _logger
import tqdm as _tqdm
import torch as _torch
import torch.utils.data as _tc_data
import torch.nn as _nn
import torch.optim as _opt
import torch.optim.lr_scheduler as _lr_sche
import torchvision.models as _tvmodels


class RunParams:
    """
    Parameters for trainning.
    1. model
    2. epoch
    3. criterion
    4. optimizer
    ...

    For easy save and easy load from file.
    """

    VERSION = "1.0.0"

    _STATE_ATTR_KEY: tuple[str] = (
        "model",
        "model",
        "optimizer",
        "criterion",
        "scheduler",
    )

    def __init__(
        self,
        model: _nn.Module,
        epoch_range: range,
        criterion: _nn.Module = None,
        optimizer: _opt.Optimizer = None,
        scheduler: _lr_sche.LRScheduler = None,
        taskname: str = "",
        arch_info: dict[str, str] = None,
        comments: dict = None,
    ) -> None:
        """
        Parameters
        ---
        model : torch.nn.Module
            model.
        epoch_range : range
            epoch range.
        criterion : torch.nn.Module, default = None
            criterion, it always be a loss function.
        optimizer : torch.optim.Optimizer, default = None
            optimizer, optional if for validate.
        scheduler: torch.optim.lr_scheduler.LRScheduler, default = None
            scheduler, optional.
        taskname : str
            name of task, for logging.
        arch_info : dict[str, str], default = {}
            Info of model, optimizer...
        comments : dict, default = {}
            comments.
        """
        self.model: _nn.Module = model
        self.epoch_range: range = epoch_range

        self.criterion: _nn.Module = criterion
        self.optimizer: _opt.Optimizer = optimizer
        self.scheduler: _lr_sche.LRScheduler = scheduler

        self.taskname: str = taskname
        self.arch_info: str = {} if arch_info is None else arch_info
        self.comments: dict = {} if comments is None else comments
        # ===
        self.reset_epoch()

    @property
    def epoch(self) -> int:
        return self.__epoch

    def next_epoch(self) -> int | None:
        try:
            self.__epoch = next(self.__epoch_iter)
            return self.__epoch
        except StopIteration:
            return None

    def reset_epoch(self):
        """
        Reset epoch and epoch_iter.
        """
        self.__epoch: int = self.epoch_range.start
        self.__epoch_iter: _typing.Iterable[int] = iter(self.epoch_range)

    def save(self, file_like: _pathlib.Path):
        """
        Save parameters to file.
        """
        def get_state_dict(item) -> None | dict[str, _typing.Any]:
            return None if item is None else item.state_dict()

        data = {
            "epoch": self.__epoch,
            "epoch_range": self.epoch_range,
            "taskname": self.taskname,
            "arch_info": self.arch_info,
            "comments": self.comments,
            "version": self.VERSION,
        }

        data.update({k: get_state_dict(self.__dict__[k]) for k in self._STATE_ATTR_KEY})

        # saving model
        _torch.save(data, file_like)

    def load(self, file_like: _pathlib.Path):
        """
        Load parameters form file.
        """
        loaded: dict = _torch.load(file_like)

        self.epoch_range: range = loaded["range"]
        self.__epoch: int = loaded["epoch"]
        self.__epoch_iter: _typing.Iterable[int] = iter(
            range(
                self.__epoch + self.epoch_range.step,
                self.epoch_range.stop,
                self.epoch_range.step,
            )
        )

        self.taskname = loaded["taskname"]
        self.arch_info = loaded["arch_info"]
        self.comments = loaded["comments"]

        # load state dict
        for k in self._STATE_ATTR_KEY:
            if loaded.get(k) is None:
                continue
            if self.__dict__[k] is None:
                # TODO, arch init
                raise RuntimeError(f"Unable to update {k}.")
            self.__dict__[k].load_state_dict(loaded[k])


class ModelHandler(_typing.Protocol):
    """
    You need to implement `iteration_forward` by you self.

    Examples
    ---
    >>> handler = ModelHandler(*args, **kwargs)
    >>> for _ in handler:
    >>>     pass
    """

    train: bool
    run_params: RunParams
    dataloader: _tc_data.DataLoader
    device: _torch.device
    callbacks: list[_typing.Callable[[_typing.Self], None]]
    __privates: dict[str, _typing.Any]

    def __init__(
        self,
        train: bool,
        run_params: RunParams,
        dataloader: _tc_data.DataLoader,
        device: _torch.device = _torch.device("cuda")
        if _torch.cuda.is_available()
        else _torch.device("cpu"),
        callbacks: list[_typing.Callable[[_typing.Self], None]] = None,
    ) -> None:
        """
        Parameters
        ---
        train : bool
            Train of Vaildate
        run_params : RunParams
            Parameters for running.
        dataloader : Dataloader
            dataloader.
        callbacks : list[(Self) -> None], default = []
            callback function, execute for each epoch at end.
        device : torch.device, default = device("cuda") | device("cpu")
            If cuda avilable, default is device("cuda") else device("cpu").
        """
        assert train in ("train", "validate")
        self.train: bool = train
        self.run_params: RunParams = run_params
        self.dataloader: _tc_data.DataLoader = dataloader
        self.device: _torch.device = device
        self.callbacks: list[_typing.Callable[[_typing.Self], None]] = (
            [] if callbacks is None else callbacks
        )
        self.__privates: dict[str, _typing.Any] = {}

    def __len__(self) -> int:
        """
        Epoch number.
        """
        return len(self.run_params.epoch_range)

    def __iter__(self) -> _typing.Self:
        """
        Do each Epoch.
        """
        return self

    def __next__(self) -> None:
        """
        Do a epoch.

        Notions
        ---
        When you call this function
        1. model will automatic switch to `train` or `eval` mode
            , and automatic `set grad enabled` or not.
        2. call method `do_epoch`.
        3. do `scheduler.step()` if in train mode.
        4. call `callbacks`.

        Raises
        ---
        StopIteration
            If run_params get the last epoch, then, raise StopIteration.
        """
        run_params: RunParams = self.run_params

        if run_params.next_epoch() is None:
            raise StopIteration("Epoch end.")

        run_params: RunParams = self.run_params
        if self.train:
            # switch to train mode
            run_params.model.train()
        else:
            # switch to evaluate mode
            run_params.model.eval()

        # do epoch
        with _torch.set_grad_enabled(self.train):
            self.do_epoch()

        # scheduler
        if run_params.scheduler is not None and self.train:
            run_params.scheduler.step()

        # callbacks
        for callback in self.callbacks:
            callback(self)

    def do_epoch(self):
        """
        Do a epoch.
        """
        for batch_data in self.dataloader:
            # batch_train
            loss = self.iteration_forward(batch_data)
            self.iteration_backward(loss)

    def iteration_forward(self, batch_data: _typing.Any) -> _torch.Tensor:
        """
        Forward part of a iteration.

        Parameters
        ---
        batch_data : Any
            self.dataloader yielded.

        Returns
        ---
        loss : Tensor
            return loss, for backward.
        """
        raise NotImplementedError("You should implemented method `iteration_forward`.")
        # example code for image classify.
        images, target = batch_data
        run_params = self.run_params

        model: _nn.Module = run_params.model
        criterion: _nn.Module = run_params.criterion
        optimizer: _opt.Optimizer = run_params.optimizer
        device: _torch.device = device

        # move data to the same device as model
        images: _torch.Tensor = images.to(device, non_blocking=True)
        target: _torch.Tensor = target.to(device, non_blocking=True)

        # compute output
        output: _torch.Tensor = model(images)
        loss: _torch.Tensor = criterion(output, target)
        return loss

    def iteration_backward(self, loss: _torch.Tensor) -> None:
        """
        Backward part of a iteration.

        Parameters
        ---
        loss : Tensor
            loss, for backward.
        """
        if self.train:
            optimizer: _opt.Optimizer = self.run_params.optimizer
            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def add_callback(self, callback: _typing.Callable[[_typing.Self], None]):
        """
        Add callback function, callback funtion will run at epoch end.

        Parameters
        ---
        callback : (Self) -> None
            callback function.
        """
        self.callbacks.append(callback)


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


class ImageClassifyHandler(ModelHandler, _typing.Protocol):
    """
    Examples
    ---
    >>> handler = CallbackHandler(*args, **kwargs)
    >>> for _ in handler:
    >>>     pass
    """

    tracers: list[RunTracer]

    def __init__(
        self,
        train: bool,
        run_params: RunParams,
        dataloader: _tc_data.DataLoader,
        device: _torch.device = _torch.device("cuda")
        if _torch.cuda.is_available()
        else _torch.device("cpu"),
        callbacks: list[_typing.Callable[[_typing.Self], None]] = None,
    ) -> None:
        super().__init__(train, run_params, dataloader, device, callbacks)
        self.tracers: list[RunTracer] = []

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
            scalar float tensor, top k accuracy.
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

    @property
    def tracer(self) -> RunTracer:
        if self.tracers:
            return self.tracers[-1]
        else:
            tracer = RunTracer(
                len(self.dataloader),
                f"{self.train} {self.run_params.taskname}: ",
            )
            self.tracers.append(tracer)
            return tracer

    @tracer.setter
    def tracer(self, new_tracer):
        self.tracers.append(new_tracer)

    def do_epoch(self):
        tracer = self.tracer
        run_params = self.run_params
        dataloader = self.dataloader

        time_anchor = _time.time()
        bar_dataloader = _tqdm.tqdm(
            dataloader, desc=f"{run_params.taskname}[{run_params.epoch}]"
        )

        for batch_data in dataloader:
            # move data to the same device as model
            images, target = batch_data
            images: _torch.Tensor = images.to(self.device, non_blocking=True)
            target: _torch.Tensor = target.to(self.device, non_blocking=True)
            tracer._load_time.add(_time.time() - time_anchor)

            # batch_train
            yield (images, target)

            # measure elapsed time
            tracer._model_time.add(_time.time() - time_anchor)
            time_anchor = _time.time()

            bar_dataloader.desc = f"{run_params.taskname}[{run_params.epoch}](acc={tracer.aver_acc1:6.4f}|loss={tracer.aver_loss:.4e})"

        tracer.log_summary()

    def iteration_forward(
        self, batch_data: tuple[_torch.Tensor, _torch.Tensor]
    ) -> _torch.Tensor:
        """
        Forward part of a iteration.

        Parameters
        ---
        batch_data : tuple[Tensor, Tensor]
            self.dataloader yielded. Images and labels.

        Returns
        ---
        loss : Tensor
            return loss, for backward.
        """
        images, target = batch_data
        run_params = self.run_params
        tracer = self.tracer

        model: _nn.Module = run_params.model
        criterion: _nn.Module = run_params.criterion
        device: _torch.device = device

        # compute output
        output = model(images)
        loss: _torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = self.accuracy(output, target, top_k=(1, 5))
        tracer._losses.add(loss.item(), images.shape[0])
        tracer._acc1.add(acc1[0], images.shape[0])
        tracer._acc5.add(acc5[0], images.shape[0])
        return loss
