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
        score: float = float("-inf"),
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
        score : float, default = float("-inf")
            score of the model. Which always positive correlation with accuracy and negetive correlation with loss.
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

        self.score: float = score
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
    _privates: dict[str, _typing.Any]

    def __init__(
        self,
        train: bool,
        run_params: RunParams,
        dataloader: _tc_data.DataLoader,
        device: _torch.device = None,
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
        self.train: bool = train
        self.run_params: RunParams = run_params
        self.dataloader: _tc_data.DataLoader = dataloader
        if device is None:
            self.device: _torch.device = (
                _torch.device("cuda")
                if _torch.cuda.is_available()
                else _torch.device("cpu")
            )
        else:
            self.device: _torch.device = device
        self.callbacks: list[_typing.Callable[[_typing.Self], None]] = (
            [] if callbacks is None else callbacks
        )
        self._privates: dict[str, _typing.Any] = {}

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

        1. get `batch_data` from `self.dataloader`.
        2. call `iteration_forward(batch_data)` for calculating loss.
        3. call `iteration_backward(loss)` for backward.
        4. update `run_params.score`.
        """
        for batch_data in self.dataloader:
            # batch_train
            loss = self.iteration_forward(batch_data)
            self.iteration_backward(loss)
            self.run_params.score = -loss.item()

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
    class MeterArray:
        def __init__(
            self,
            *,
            alias="",
            fmt=":.4e",
            summary_type: _typing.Literal["avg", "sum"] = "avg",
        ) -> None:
            self.__values: list[float] = []
            self.__weights: list[float] = []
            self.fmt = fmt
            self.alias = alias
            self.summary_type = summary_type

        def add(self, value: float | tuple[float, float]):
            """
            Parameters
            ---
            value : float | tuple[float, float]
                value or (value, weight)
            """
            if isinstance(value, float):
                self.__values.append(float(value))
                self.__weights.append(1)
            elif isinstance(value, tuple) and len(value) == 2:
                value, weight = value
                self.__values.append(float(value))
                self.__weights.append(float(weight))
            else:
                raise ValueError(value)
            return self

        def __iadd__(self, value: float | tuple[float, float]):
            """
            Parameters
            ---
            value : float | tuple[float, float]
                value or (value, weight)
            """
            return self.add(value)

        @property
        def datas(self) -> dict[_typing.Literal["values", "weights"], list[float]]:
            return {"values": self.__values.copy(), "weights": self.__weights.copy()}

        @property
        def avg(self) -> float:
            return self.sum / sum(self.__weights)

        @property
        def sum(self) -> float:
            return sum((i * j for i, j in zip(self.__values, self.__weights)))

        @property
        def value(self) -> float | None:
            if self.__values:
                return self.__values[-1]
            return None

        def __str__(
            self,
        ) -> str:
            """
            Returns
            ---
            str :
                {alias}={value:fmt}({summary:fmt})

            Examples
            ---
            >>> ma = MeterArray(alias="loss", fmt=":.4e", summary_type="avg")
            >>> ma += 1.2  # add new value to loss, weight = 1
            >>> ma += (3.4, 5.6)  # add new value to loss, weight = 5.6
            >>> print(ma.avg)  # (1.2*1 + 3.4*5.6) / (1+5.6) = 3.067
            <<< 3.0666666666666664
            >>> print(ma)
            <<< loss=3.4000e+00(3.0667e+00)
            """
            fmtstr = "{alias}={val%s}({summary%s})" % (self.fmt, self.fmt)
            match self.summary_type:
                case "avg":
                    meter_summary = self.avg
                case "sum":
                    meter_summary = self.sum

            return fmtstr.format(
                **{"alias": self.alias, "val": self.value, "summary": meter_summary}
            )

    meters: dict[str, MeterArray]

    def __init__(
        self,
    ) -> None:
        self.meters = {}

    def __getitem__(self, key: str) -> MeterArray:
        """
        MeterArray have been implemented `__iadd__` method.

        Examples
        ---
        >>> m = Meter()
        >>> m["loss"] += 1.2  # add new value to loss, weight = 1
        >>> m["loss"] += (3.4, 5.6)  # add new value to loss, weight = 5.6
        >>> print(m["loss"].avg)  # (1.2*1 + 3.4*5.6) / (1+5.6) = 3.067
        <<< 3.0666666666666664
        """
        if key not in self.meters:
            self.meters[key] = self.MeterArray(alias=key)
        return self.meters[key]

    def __setitem__(self, key: str, value: MeterArray):
        self.meters[key] = value

    def keys(self):
        return self.meters.keys()

    def values(self):
        return self.meters.values()

    def items(self):
        return self.meters.items()

    def as_dict(
        self,
    ) -> dict[str, dict[_typing.Literal["values", "weights"], list[float]]]:
        """
        Return all the data in meter_array.
        """
        return {k: v.datas for k, v in self.meters.items()}

    def add_meter(
        self,
        key: str,
        *,
        alias=None,
        fmt=":.4e",
        summary_type: _typing.Literal["avg", "sum"] = "avg",
    ):
        alias = key if alias is None else alias
        self.meters[key] = self.MeterArray(
            alias=alias,
            fmt=fmt,
            summary_type=summary_type,
        )

    def __str__(self) -> str:
        return "\t".join(map(str, self.meters.values()))


class ImageClassifyHandler(ModelHandler, _typing.Protocol):
    """
    Examples
    ---
    >>> handler = CallbackHandler(*args, **kwargs)
    >>> for _ in handler:
    >>>     pass
    """

    meters: list[Meter]

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
        self.meters: list[Meter] = []

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
    def meter(self) -> Meter:
        key = "meter"
        if key not in self._privates:
            new_meter = Meter()
            self._privates[key] = new_meter
            self.meters.append(new_meter)
        return self._privates[key]

    @meter.setter
    def meter(self, new_meter):
        key = "meter"
        self._privates[key] = new_meter
        self.meters.append(new_meter)

    def do_epoch(self):
        self.meter = Meter()
        meter = self.meter

        meter.add_meter("model", fmt=":6.3f")  # time of model dealing the Tensor
        meter.add_meter("load", fmt=":6.3f")  # time of dataloader load the data
        meter.add_meter("loss", fmt=":.4e")  # loss
        meter.add_meter("acc1", fmt=":6.2f")  # accuracy 1
        meter.add_meter("acc5", fmt=":6.2f")  # accuracy 5

        run_params = self.run_params
        dataloader = self.dataloader

        time_anchor = _time.time()
        bar_dataloader = _tqdm.tqdm(
            dataloader, desc=f"{run_params.taskname}[{run_params.epoch}]"
        )

        for batch_data in bar_dataloader:
            meter["load"] += _time.time() - time_anchor

            # batch_train
            loss = self.iteration_forward(batch_data)
            self.iteration_backward(loss)

            # measure elapsed time
            meter["model"] += _time.time() - time_anchor
            time_anchor = _time.time()

            bar_dataloader.desc = f"{run_params.taskname}[{run_params.epoch}](acc={meter['acc1'].avg:6.4f}|loss={meter['loss'].avg:.4e})"

            self.run_params.score = meter["acc1"].avg

        _logger.debug(f"[{self.run_params.epoch}] {meter}")

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
        # move data to the same device as model
        images, target = batch_data
        images: _torch.Tensor = images.to(self.device, non_blocking=True)
        target: _torch.Tensor = target.to(self.device, non_blocking=True)
        run_params = self.run_params
        meter = self.meter

        model: _nn.Module = run_params.model
        criterion: _nn.Module = run_params.criterion

        # compute output
        output = model(images)
        loss: _torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = self.accuracy(output, target, top_k=(1, 5))
        meter["loss"] += (loss.item(), images.shape[0])
        meter["acc1"] += (acc1[0], images.shape[0])
        meter["acc5"] += (acc5[0], images.shape[0])
        return loss
