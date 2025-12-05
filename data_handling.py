import pandas as pd
from datasets import Dataset

# Define your label mappings
label_list = ["O", "B-Object", "I-Object", "B-Aspect", "I-Aspect", "B-Predicate", "I-Predicate"]
label_encoding = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

def read_conll_file(file_path):
    """
    Reads the CoNLL file and groups words/labels by sentence.
    """
    data = {'tokens': [], 'ner_tags': []}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                # Empty line indicates end of sentence
                if words:
                    data['tokens'].append(words)
                    data['ner_tags'].append(labels)
                    words = []
                    labels = []
            else:
                parts = line.split('\t') # PDF says tab separated
                if len(parts) == 2:
                    words.append(parts[0])
                    # Map text label (e.g., 'B-Object') to ID (e.g., 1)
                    labels.append(label_encoding.get(parts[1], 0)) 
                    
        # Catch the last sentence if no newline at very end
        if words:
            data['tokens'].append(words)
            data['ner_tags'].append(labels)
            
    return Dataset.from_dict(data)

# Usage (Upload your .txt files to Colab first)
# train_dataset = read_conll_file("train.txt")
# val_dataset = read_conll_file("dev.txt") # Assuming you split or have a dev set