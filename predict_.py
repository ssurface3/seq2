import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import input_data

# --- SETTINGS ---
# Use the folder created by train_advanced.py
MODEL_PATH = "deberta-optimized-model" 
TEST_FILE = "test_no_answers.tsv"
OUTPUT_FILE = "submission_deberta.tsv"

def predict():
    print(f"Loading DeBERTa model from {MODEL_PATH}...")
    
    # 1. Load Tokenizer & Model
    # We load the tokenizer base to ensure we have the sentencepiece files
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Read Test Data
    raw_data = input_data.read_conll_file(TEST_FILE)
    sentences = raw_data['tokens']

    print(f"Predicting on {len(sentences)} sentences...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for sentence_tokens in sentences:
            tokenized_inputs = tokenizer(
                sentence_tokens,
                truncation=True,
                is_split_into_words=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**tokenized_inputs)
            
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
            word_ids = tokenized_inputs.word_ids()
            
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx, pred_id in zip(word_ids, predictions):
                if word_idx is None:
                    continue
                if word_idx != previous_word_idx:
                    label_str = input_data.ID2LABEL[pred_id]
                    aligned_labels.append(label_str)
                    previous_word_idx = word_idx
            
            # Fill logic
            while len(aligned_labels) < len(sentence_tokens):
                aligned_labels.append("O")
            aligned_labels = aligned_labels[:len(sentence_tokens)]

            for word, label in zip(sentence_tokens, aligned_labels):
                f_out.write(f"{word}\t{label}\n")
            f_out.write("\n")

    print(f"Done! Submission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    predict()