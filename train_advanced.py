import torch
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoTokenizer
)
import evaluate
import numpy as np
import input_data  
import optuna


MODEL_CHECKPOINT = "microsoft/deberta-v3-large" 
TRAIN_FILE = "train.tsv"
OUTPUT_DIR = "deberta-optimized-model"


def get_class_weights(tokenized_dataset):
    """
    Calculates weights based on Inverse Class Frequency.
    Rare classes (Aspects) get higher weights.
    """
    labels = [label for sublist in tokenized_dataset['labels'] for label in sublist if label != -100]
    classes, counts = np.unique(labels, return_counts=True)
    

    weights = 1.0 / (counts / len(labels))
    

    weights = weights / weights[0] 
    
    weights[3] *= 2.0  
    weights[4] *= 2.0  
    
    print("Calculated Class Weights:", weights)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        device = inputs["input_ids"].device
        

        if self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)

   
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        
   
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        else:
            return outputs

seqeval = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [input_data.ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [input_data.ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    print("Loading Data...")
    tokenized_datasets, tokenizer = input_data.get_tokenized_dataset(TRAIN_FILE)
    
    class_weights = get_class_weights(tokenized_datasets["train"])

    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=len(input_data.LABEL_LIST),
            id2label=input_data.ID2LABEL,
            label2id=input_data.LABEL2ID
        )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 6)
        }

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1, 
        push_to_hub=False,
        disable_tqdm=False 
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model_init=model_init,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", tokenized_datasets.get("test")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting Hyperparameter Search...")
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=hp_space,
        n_trials=10,  
        backend="optuna"
    )

    print(f"Best Run Found: {best_run}")

    
    print("Training final model with best parameters...")
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train()
    
    print(f"Saving Final Optimized Model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()