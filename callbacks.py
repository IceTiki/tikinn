import matplotlib.pyplot as _plt
import pathlib as _pathlib
import shutil as _shutil
import json as _json

import torch as _torch

from .driver import RunHandler
from .constants import Directions


class AutoPlot:
    def __init__(self, output_folder: _pathlib.Path, with_data=True) -> None:
        self.output_folder = _pathlib.Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.with_data = with_data

    @staticmethod
    def plot_at_ax(
        ax: _plt.Axes,
        x: list[float],
        y: list[float],
        x_label="",
        y_label="",
        title="",
    ):
        ax.plot(x, y)
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

    def __call__(self, run_handler: RunHandler):
        x = [
            i for i, _ in zip(run_handler.epoch_range, range(len(run_handler.tracers)))
        ]
        y1 = [i.aver_acc1 for i in run_handler.tracers]
        y2 = [i.aver_loss for i in run_handler.tracers]

        fig, axs = _plt.subplots(
            nrows=1, ncols=2, figsize=(18, 12), constrained_layout=True
        )
        ax1, ax2 = axs
        ax1: _plt.Axes
        ax2: _plt.Axes

        ax1.set_ylim(0, 100)

        self.plot_at_ax(ax1, x, y1, "epoch", "acc1", f"acc1 of {run_handler.name}")
        self.plot_at_ax(ax2, x, y2, "epoch", "loss", f"loss of {run_handler.name}")

        output_folder = self.output_folder
        img_out = output_folder / f"{run_handler.name}.png"
        json_out = output_folder / f"{run_handler.name}.json"

        _plt.savefig(img_out, dpi=150)
        _plt.close()

        if self.with_data:
            with open(json_out, "w", encoding="utf-8") as f:
                _json.dump(
                    {
                        "epoch": x,
                        "acc1": y1,
                        "loss": y2,
                        "full": run_handler.tracer_data,
                    },
                    f,
                    ensure_ascii=False,
                )


class AutoSave:
    def __init__(self, save_folder: _pathlib.Path, arch: str, best_copy=None) -> None:
        self.save_folder = _pathlib.Path(save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0
        self.arch = arch
        self.best_copy = best_copy

    @staticmethod
    def save_checkpoint(state, is_best, filepath="checkpoint.pth.tar"):
        filepath = _pathlib.Path(filepath)
        _torch.save(state, filepath)
        if is_best:
            _shutil.copyfile(filepath, filepath.parent / "model_best.pth.tar")

    def __call__(self, run_handler: RunHandler):
        acc = run_handler.last_tracer.aver_acc1
        if is_best := (acc > self.best_acc):
            self.best_acc = acc

        savedir = self.save_folder / f"{run_handler.epoch}.pth.tar"
        run_handler.save(savedir)
        if is_best:
            _shutil.copyfile(savedir, self.save_folder / f"best.pth.tar")
            if self.best_copy is not None:
                _shutil.copyfile(savedir, self.best_copy)
