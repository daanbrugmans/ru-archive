from torch import nn
from peft import LoraConfig, get_peft_model
from models.bert import BERTClassifier

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class PEFTBERTClassifier(BERTClassifier):
    def __init__(self, writer, bert_model_name, num_classes, best_model_output_path, lora_config):
        BERTClassifier.__init__(self, writer=writer, bert_model_name=bert_model_name, num_classes=num_classes, best_model_output_path=best_model_output_path, lora_config=None)
        config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 16),
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["classifier"],
        )
       
        self.bert = get_peft_model(self.bert, peft_config=config)
        print_trainable_parameters(self.bert)
