from sentence_transformers.evaluation import SentenceEvaluator
import torch
from torchmetrics import F1Score, Precision, Recall
from torch.utils.data import DataLoader
import logging
from sentence_transformers.util import batch_to_device


logger = logging.getLogger(__name__)

class F1Evaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, writer, dataloader: DataLoader, name: str = "", softmax_model = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "f1_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "f1"]

        self.writer = writer

        self.f1_obj = F1Score(task="multiclass", num_classes=3)
        self.precision_obj = Precision(task="multiclass", num_classes=3, average="weighted")
        self.recall_obj = Recall(task="multiclass", num_classes=3, average="weighted")

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate

        model_outputs = []
        all_label_ids = []

        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            all_label_ids.append(label_ids.cpu())
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)
                model_outputs.append(prediction.cpu())
        
        all_predictions = torch.cat(model_outputs, dim=0)
        all_label_ids = torch.cat(all_label_ids, dim=0)

        f1_score = self.f1_obj(all_predictions, all_label_ids)
        precision = self.precision_obj(all_predictions, all_label_ids)
        recall = self.recall_obj(all_predictions, all_label_ids)

        logger.info(f"F1 score: {f1_score}")

        self.writer.add_scalar('Validation/F1-score', f1_score, steps)
        self.writer.add_scalar('Validation/Precision', precision, steps)
        self.writer.add_scalar('Validation/Recall', recall, steps)

        return f1_score
