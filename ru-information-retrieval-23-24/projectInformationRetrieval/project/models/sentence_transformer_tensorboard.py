import os
from typing import Iterable, Optional, Union

from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class SentenceTransformerTensorboard(SentenceTransformer):
    def __init__(
		self, 
        writer: SummaryWriter,
        best_model_output_path: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
		modules: Optional[Iterable[nn.Module]] = None,
		device: Optional[str] = None,
		cache_folder: Optional[str] = None,
		use_auth_token: Union[bool, str, None] = None,
	):
        super().__init__(model_name_or_path, modules, device, cache_folder, use_auth_token)
        self.writer = writer
        self.best_model_output_path = best_model_output_path

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
		# super._eval_during_training(evaluator, output_path, save_best_model, epoch, steps, callback)
        f1_score = evaluator(self, epoch=epoch, steps=steps)
        if callback is not None:
            callback(f1_score, epoch, steps)
        if f1_score > self.best_score:
            self.best_score = f1_score
            if save_best_model:
                self.save(self.best_model_output_path)
