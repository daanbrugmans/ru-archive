import os
import torch
import random
import json
import numpy as np
from models.model_handler import ModelHandler
from data_loading.data_loader import CovidDataLoader
from models.peft_bert import PEFTBERTClassifier
from models.bert import BERTClassifier


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

data_dir = os.path.join(os.getcwd(), "project", "data")
trec_covid_csv_path = os.path.join(data_dir, "trec_covid_beir.csv")
# trec_covid_csv_path = os.path.join(data_dir, "dummy_data.csv")

dataPreprocessor = CovidDataLoader()
dataPreprocessor.load_data(trec_covid_csv_path=trec_covid_csv_path)

train_dataloader_unbalanced, val_dataloader_unbalanced, test_dataloader_unbalanced = dataPreprocessor.make_dataloaders(shuffle=True, batch_size=32, train=0.6, val=0.2, test=0.2)
dataPreprocessor.balance_classes_by_removing_samples()
balanced_train_dataloader, balanced_val_dataloader, balanced_test_dataloader = dataPreprocessor.make_dataloaders(shuffle=True, batch_size=32, train=0.6, val=0.2, test=0.2, balanced = True)

with open("project/config_part2.json", "r") as file:
    training_configs = json.load(file)

for training_config in training_configs:
    if training_config.get('balance', True):
        # use balanced data
        train_dataloader = balanced_train_dataloader
        val_dataloader = balanced_val_dataloader
        test_dataloader = balanced_test_dataloader
    else:
        # use unbalanced data
        train_dataloader = train_dataloader_unbalanced
        val_dataloader = val_dataloader_unbalanced
        test_dataloader = test_dataloader_unbalanced

    run_name = f"{training_config.get('name')}_lr:{training_config['learning_rate']}_e:{training_config['num_epochs']}_s:{training_config['scheduler']}"

    if training_config.get("name") == "bert":
        bert = ModelHandler(BERTClassifier, run_name=run_name, best_model_output_path='saved_models/tuned_bert_model')
    elif training_config.get("name") == "peft_bert":
        run_name = run_name + "_lora:" + str(training_config["lora_config"]["r"])
        bert = ModelHandler(PEFTBERTClassifier, run_name=run_name, best_model_output_path='saved_models/tuned_peft_model', lora_config=training_config['lora_config'])
    else:
        raise ValueError(f"name '{training_config['name']}' in config not allowed.")

    print("Training peft model")
    bert.train(
        train_dataloader=train_dataloader, 
        val_evaluation_steps=100,
        learning_rate=training_config["learning_rate"], 
        num_epochs=training_config["num_epochs"], 
        val_dataloader=val_dataloader,
        scheduler=training_config["scheduler"],
    )
    torch.save(bert.model, f'saved_models/{run_name}')

    train_scores = bert.evaluate(train_dataloader)
    val_scores = bert.evaluate(val_dataloader)
    test_scores = bert.evaluate(test_dataloader)

    with open(f"output/output_scores_{run_name}.json", "w") as file:
        json.dump({
            "config": training_config,
            "train_scores": train_scores,
            "val_scores": val_scores,
            "test_scores": test_scores,
            },
            file
        )
