from typing import Callable, Dict, Iterable
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.losses import SoftmaxLoss
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter


class SoftmaxLossTensorboard(SoftmaxLoss):
	def __init__(
		self,
		writer: SummaryWriter,
		model: SentenceTransformer,
		sentence_embedding_dimension: int,
		num_labels: int,
		concatenation_sent_rep: bool = True,
		concatenation_sent_difference: bool = True,
		concatenation_sent_multiplication: bool = False,
		loss_fct: Callable = nn.CrossEntropyLoss()
	):
		super().__init__(model, sentence_embedding_dimension, num_labels, concatenation_sent_rep, concatenation_sent_difference, concatenation_sent_multiplication, loss_fct)
		self.writer = writer
		self.loss_steps = 0

	def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
		forward_super_output = super().forward(sentence_features, labels)
		if labels is not None:
			loss = forward_super_output
			self.writer.add_scalar('Train/Loss', float(loss.detach().cpu()), self.loss_steps)
			self.loss_steps += 1

		return forward_super_output 