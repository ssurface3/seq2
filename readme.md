***

# Advanced Named Entity Recognition with DeBERTa V3 & Optuna

This project implements a high-performance Named Entity Recognition (NER) pipeline designed to extract **Objects**, **Aspects**, and **Predicates** from text. It utilizes **Microsoft's DeBERTa V3** architecture, enhanced with custom loss weighting for class imbalance and **Optuna** for automated hyperparameter optimization.

## Features

*   **Model**: Uses `microsoft/deberta-v3-large` for contextual embedding and token classification.
*   **Hyperparameter Search**: Integrated **Optuna** backend to automatically find the best learning rate, batch size, weight decay, and epoch count.
*   **Imbalance Handling**: Implements a `WeightedTrainer` that calculates **Inverse Class Frequency** to penalize the model more for missing rare tags (specifically boosting Aspect detection).
*   **robust Data Processing**: Handles CoNLL format data, sub-word tokenization alignment, and sequence evaluation.

## Project Structure

*   `train_advanced.py`: Main training script. Handles class weight calculation, Optuna integration, and the training loop.
*   `predict_.py`: Inference script. Loads the optimized model to generate predictions on test data without labels.
*   `input_data.py`: Utilities for loading CoNLL datasets, tokenizing with DeBERTa, and aligning labels with sub-word tokens.
*   `data_handling.py`: Helper script for reading raw CoNLL files into HuggingFace Dataset objects.

## Installation
Requirements for the code runningg
```bash
pip install -r requirements.txt
```

## Data Format

The project expects data in **CoNLL format** (Tab-Separated Values).

### Training Data (`train.tsv`)
Two columns: `Word` and `Label`.
```text
The	O
camera	B-Object
quality	B-Aspect
is	O
amazing	B-Predicate
```

**Supported Labels:**
*   `O` (Outside)
*   `B-Object`, `I-Object`
*   `B-Aspect`, `I-Aspect`
*   `B-Predicate`, `I-Predicate`

### Test Data (`test_no_answers.tsv`)
One column (or just words separated by newlines/tabs) containing the raw tokens to predict.

## Usage

### 1. Training the Model
To start the training process, run the advanced training script. This will:
1. Load `train.tsv`.
2. Calculate class weights to handle label imbalance.
3. Run an Optuna hyperparameter search (10 trials).
4. Retrain the final model with the best parameters.
5. Save the model to `./deberta-optimized-model`.

```bash
python train_advanced.py
```

### 2. Inference (Prediction)
Once training is complete, generate predictions for your test set. This script reads `test_no_answers.tsv` and generates `submission_deberta.tsv`.

```bash
python predict_.py
```

## ⚙️ Advanced Configuration

### Modifying the Search Space
You can adjust the hyperparameter search space in `train_advanced.py` inside the `hp_space` function:

```python
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        # ... add more parameters here
    }
```

### Custom Class Weights
The model specifically targets improving "Aspect" detection. In `train_advanced.py`, weights are dynamically calculated, but specific classes are manually boosted:

```python
# specific boost for Aspect tags (indices 3 and 4)
weights[3] *= 2.0  
weights[4] *= 2.0  
```

## Metrics
The model is evaluated using the `seqeval` library, focusing on:
*   **Overall F1 Score** (Primary metric for optimization)
*   **Accuracy**

## Output Format
The `predict_.py` script generates a submission file compatible with CoNLL evaluation scripts:

```text
word1    predicted_label
word2    predicted_label
...
```

    # how to use it?
    chmod +x run_pipeline.sh </br>
    ./run_pipeline.sh