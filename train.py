import numpy as np
import evaluate
from transformers import (
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
import input_data  # Importing our helper file

# --- SETTINGS ---
TRAIN_FILE = "train.tsv" 
# If you have a separate dev file, change this. Otherwise it splits train automatically.
VAL_FILE = None 
# OUTPUT_DIR = "roberta-srl-model"
MODEL_CHECKPOINT = "microsoft/deberta-v3-large" 
OUTPUT_DIR = "microsoft/deberta-v3-large" 
# Or if you have a GPU with >12GB VRAM, try "microsoft/deberta-v3-large"

def main():
    # 1. Prepare Data
    print("Loading and tokenizing data...")
    tokenized_datasets, tokenizer = input_data.get_tokenized_dataset(TRAIN_FILE, VAL_FILE)
    
    # 2. Load Model
    print("Loading Model...")
    model = AutoModelForTokenClassification.from_pretrained(
        input_data.MODEL_CHECKPOINT,
        num_labels=len(input_data.LABEL_LIST),
        id2label=input_data.ID2LABEL,
        label2id=input_data.LABEL2ID
    )
    
    # 3. Define Metrics
    seqeval = evaluate.load("seqeval")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [input_data.LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [input_data.LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # 4. Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2, # Only keep the last 2 checkpoints to save space
        push_to_hub=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Run Training
    print("Starting training...")
    trainer.train()
    
    # 6. Save Final Model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()