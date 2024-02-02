import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from .sentence_transformer_tensorboard import SentenceTransformerTensorboard


class BERTClassifier(nn.Module):
    def __init__(
            self, 
            bert_model_name, 
            num_classes, 
            writer,
            best_model_output_path,
            lora_config,
            concatenation_args: dict = {
                "concatenation_sent_rep": True,
                "concatenation_sent_difference": True,
                "concatenation_sent_multiplication": False,
            }
    ):
        super(BERTClassifier, self).__init__()
        self.bert = SentenceTransformerTensorboard(model_name_or_path=bert_model_name, writer=writer, best_model_output_path=best_model_output_path)
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = self.bert[0].auto_model.base_model.config.hidden_size
        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.concatenation_args: dict = concatenation_args
        num_vectors_concatenated = 0
        if concatenation_args["concatenation_sent_rep"]:
            num_vectors_concatenated += 2
        if concatenation_args["concatenation_sent_difference"]:
            num_vectors_concatenated += 1
        if concatenation_args["concatenation_sent_multiplication"]:
            num_vectors_concatenated += 1
        self.sent_embed_linear = nn.Linear(num_vectors_concatenated * self.hidden_size, num_classes)

        if lora_config is not None:
            raise ValueError("Normal bert model got an unexpected lora config ):")

    def forward(self, *args, **kwargs):
        if "labels" in kwargs:
            outputs, logits = self.get_sentence_embedding(*args, **kwargs)
            return outputs, logits
        else:
            outputs = self.bert(*args, **kwargs)
            sentence_embedding = outputs["sentence_embedding"]
            x = self.dropout(sentence_embedding)
            logits = self.fc(x)
            return outputs

    def get_sentence_embedding(self, *args, **kwargs):
        labels = kwargs.pop("labels", None)

        outputs = [self.bert(inputs) for inputs in args[0]]
        rep_a, rep_b = [rep['sentence_embedding'] for rep in outputs]

        vectors_concat = []
        if self.concatenation_args["concatenation_sent_rep"]:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_args["concatenation_sent_difference"]:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_args["concatenation_sent_multiplication"]:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        logits = self.sent_embed_linear(features)

        return outputs, logits
