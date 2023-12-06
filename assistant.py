import pathlib as _pathlib
import shutil as _shutil
import json as _json
import itertools as _itertools
import math as _math

import matplotlib.pyplot as _plt
import numpy as _np

from .handler import ModelHandler, ImageClassifyHandler


class AutoPlot:
    def __init__(self, output_folder: _pathlib.Path) -> None:
        self.output_folder = _pathlib.Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def plot_at_ax(
        ax: _plt.Axes,
        x: list[float],
        y: list[float],
        x_label="",
        y_label="",
        title="",
        line_label="",
        alpha=1,
    ):
        if line_label:
            ax.plot(x, y, label=line_label, alpha=alpha)
        else:
            ax.plot(x, y, alpha=alpha)

        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)

        max_value = max(y)
        max_epoch = x[y.index(max_value)]
        ax.text(
            max_epoch,
            max_value,
            f"{(round(max_epoch, 4), round(max_value, 4))}",
        )

        min_value = min(y)
        min_epoch = x[y.index(min_value)]
        ax.text(
            min_epoch,
            min_value,
            f"{(round(min_epoch, 4), round(min_value, 4))}",
        )

        if line_label:
            ax.legend()

    def __call__(self, handler: ImageClassifyHandler):
        if not handler.meters:
            return

        run_params = handler.run_params
        first_epoch = run_params.epoch_range.start
        last_epoch = run_params.epoch
        epoch_step = run_params.epoch_range.step

        def plot_meter(key: str, ax: _plt.Axes):
            datas = [i.as_dict() for i in handler.meters]

            values: list[list[float]] = [i[key]["values"] for i in datas]
            weights: list[list[float]] = [i[key]["weights"] for i in datas]

            x_avg = _np.linspace(
                first_epoch + epoch_step / 2, last_epoch + epoch_step / 2
            )
            y_avg = [sum(i) / sum(j) for i, j in zip(values, weights)]

            self.plot_at_ax(
                ax,
                x_avg,
                y_avg,
                x_label="epoch",
                y_label=key,
                title=f"{key} of {run_params.taskname}",
            )

            sum_weights_accum: list[float] = [0] + list(
                _itertools.accumulate(map(sum, weights))
            )
            flat_weights = [j for i in weights for j in i]
            flat_values = [j for i in values for j in i]

            x_full = _np.interp(
                list(_itertools.accumulate(flat_weights)),
                sum_weights_accum,
                _np.arange(first_epoch, last_epoch + epoch_step, epoch_step),
            )
            y_full = flat_values

            self.plot_at_ax(
                ax,
                x_full,
                y_full,
                x_label="epoch",
                y_label=key,
                title=f"{key} of {run_params.taskname}",
                alpha=0.2,
            )

        keys = list(handler.meter.keys())

        fig_width, fig_high = 18, 12
        figsize = (fig_width, fig_high)
        figarea = fig_width * fig_high
        nrows = _math.ceil(len(keys) / figarea * fig_high)
        ncols = _math.ceil(len(keys) / figarea * fig_width)

        assert nrows * ncols >= len(keys)

        fig, axs = _plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True
        )

        for key, ax in zip(keys, _np.nditer(axs)):
            plot_meter(key, ax)

        output_folder = self.output_folder
        img_out = output_folder / f"{run_params.taskname}.png"

        _plt.savefig(img_out, dpi=200)
        _plt.close()


class AutoSave:
    def __init__(self, save_folder: _pathlib.Path) -> None:
        self.save_folder = _pathlib.Path(save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.best_score = float("-inf")

    def __call__(self, handler: ModelHandler):
        run_params = handler.run_params

        savedir = self.save_folder / f"{run_params.epoch}.pth.tar"
        run_params.save(savedir)

        is_best = run_params.score >= self.best_score
        if is_best:
            self.best_score = run_params.score
            bestdir = self.save_folder / f"best.pth.tar"
            _shutil.copyfile(savedir, bestdir)
