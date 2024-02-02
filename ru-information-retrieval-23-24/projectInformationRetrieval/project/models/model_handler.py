import torch
import os
from torch import nn
from transformers import BertTokenizer
from torchmetrics import F1Score, Precision, Recall, ConfusionMatrix
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .softmax_loss_tensorboard import SoftmaxLossTensorboard
from evaluators.f1_evaluator import F1Evaluator
from models.peft_bert import PEFTBERTClassifier
from models.bert import BERTClassifier

class ModelHandler:
    def __init__(
        self,
        classifier: nn.Module,
        device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
        model_name: str = "sentence-transformers/msmarco-bert-base-dot-v5",
        best_model_output_path: str = None,
        run_name: str = "default",
        num_classes: int = 3,
        max_length: int = 128,
        freeze=True,
        lora_config=None
    ) -> None:
        self.bert_model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.writer = SummaryWriter(f'runs/{model_name}/{run_name}')

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.device = device
        self.model: BERTClassifier | PEFTBERTClassifier = classifier(bert_model_name=self.bert_model_name, num_classes=num_classes, writer=self.writer, best_model_output_path=best_model_output_path, lora_config=lora_config).to(self.device)

        if freeze:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def train(
        self,
        train_dataloader: DataLoader,
        learning_rate: float = 2e-5,
        val_evaluation_steps: int = 500,
        num_epochs: int = 4,
        val_dataloader = DataLoader | None,
        save_best_model: bool = False,
        scheduler="constantlr",
    ):
        train_loss = SoftmaxLossTensorboard(
            model=self.model, 
            writer=self.writer,
            num_labels=self.num_classes, 
            sentence_embedding_dimension=self.model.hidden_size,
            concatenation_sent_rep=self.model.concatenation_args["concatenation_sent_rep"],
            concatenation_sent_difference=self.model.concatenation_args["concatenation_sent_difference"],
            concatenation_sent_multiplication=self.model.concatenation_args["concatenation_sent_multiplication"])        
        
        val_evaluator = F1Evaluator(writer=self.writer, dataloader=val_dataloader, name='val', softmax_model=self.model)

        self.model.bert.fit(
            train_objectives=[(train_dataloader, train_loss)], 
            epochs=num_epochs,
            evaluation_steps=val_evaluation_steps,
            optimizer_params={"lr": learning_rate},
            evaluator=val_evaluator,
            save_best_model=save_best_model,
            scheduler=scheduler,
		)

        self.writer.close()

    def evaluate(self, dataloader):
        f1_obj = F1Score(task="multiclass", num_classes=3)
        precision_obj = Precision(task="multiclass", num_classes=3, average="weighted")
        recall_obj = Recall(task="multiclass", num_classes=3, average="weighted")
        confmat_obj = ConfusionMatrix(task="multiclass", num_classes=3)
        
        self.model.bert.eval()

        dataloader.collate_fn = self.model.bert.smart_batching_collate

        model_outputs = []
        all_label_ids = []

        for step, batch in enumerate(dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], self.model.bert.device)
            label_ids = label_ids.to(self.model.bert.device)
            all_label_ids.append(label_ids.cpu())
            with torch.no_grad():
                _, prediction = self.model(features, labels=None)
                model_outputs.append(prediction.cpu())
        
        all_predictions = torch.cat(model_outputs, dim=0)
        all_label_ids = torch.cat(all_label_ids, dim=0)

        f1_score = f1_obj(all_predictions, all_label_ids)
        precision = precision_obj(all_predictions, all_label_ids)
        recall = recall_obj(all_predictions, all_label_ids)
        confmat = confmat_obj(all_predictions, all_label_ids)

        return {
            "f1_score": float(f1_score),
            "precision": float(precision),
            "recall": float(recall),
            "confmat": confmat.tolist()
        }
