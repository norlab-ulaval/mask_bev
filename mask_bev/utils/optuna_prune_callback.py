import warnings

import optuna
import pytorch_lightning as pl


# Inspired by https://optuna.readthedocs.io/en/stable/_modules/optuna/integration/pytorch_lightning.html
class OptunaPruneCallback(pl.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str):
        self._trial = trial
        self._monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self._monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self._monitor)
            )
            warnings.warn(message)
            return

        should_stop = False
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()
        should_stop = trainer.training_type_plugin.broadcast(should_stop)
        if not should_stop:
            return

        message = "Trial was pruned at epoch {}.".format(epoch)
        raise optuna.TrialPruned(message)
