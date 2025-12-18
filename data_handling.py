import pandas as pd
from datasets import Dataset

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
                if words:
                    data['tokens'].append(words)
                    data['ner_tags'].append(labels)
                    words = []
                    labels = []
            else:
                parts = line.split('\t') 
                if len(parts) == 2:
                    words.append(parts[0])
                    labels.append(label_encoding.get(parts[1], 0)) 
                
        if words:
            data['tokens'].append(words)
            data['ner_tags'].append(labels)
            
    return Dataset.from_dict(data)

