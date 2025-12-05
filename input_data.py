import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# --- CONFIGURATION (UPDATED TO DEBERTA) ---
# We use DeBERTa V3 Base. 
# Note: You must run "pip install sentencepiece" for this to work.
MODEL_CHECKPOINT = "microsoft/deberta-v3-base" 

LABEL_LIST = ["O", "B-Object", "I-Object", "B-Aspect", "I-Aspect", "B-Predicate", "I-Predicate"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

def read_conll_file(file_path):
    data = {'tokens': [], 'ner_tags': []}
    current_tokens = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    data['tokens'].append(current_tokens)
                    data['ner_tags'].append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split('\t')
                current_tokens.append(parts[0])
                if len(parts) > 1:
                    current_labels.append(LABEL2ID.get(parts[1], 0))
                else:
                    current_labels.append(0)
                    
        if current_tokens:
            data['tokens'].append(current_tokens)
            data['ner_tags'].append(current_labels)
    return data

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            new_labels.append(-100)
        elif word_idx != previous_word_idx:
            new_labels.append(labels[word_idx])
        else:
            new_labels.append(-100)
        previous_word_idx = word_idx
    return new_labels

def tokenize_and_align(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels.append(align_labels_with_tokens(label, word_ids))
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def get_tokenized_dataset(train_path, val_path=None):
    # DEBERTA TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    train_data = read_conll_file(train_path)
    hf_train = Dataset.from_dict(train_data)
    
    dataset_dict = {'train': hf_train}
    
    if val_path:
        val_data = read_conll_file(val_path)
        hf_val = Dataset.from_dict(val_data)
        dataset_dict['validation'] = hf_val
    else:
        split = hf_train.train_test_split(test_size=0.1)
        dataset_dict = split
        
    raw_datasets = DatasetDict(dataset_dict)
    
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align(x, tokenizer),
        batched=True
    )
    
    return tokenized_datasets, tokenizer