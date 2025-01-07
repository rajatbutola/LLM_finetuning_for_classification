# LLM_finetuning_for_classification
Fine-tuning a BERT model for sentence-pair classification dataset using Hugging Face Transformers.
# Fine-Tuning BERT  

This repository demonstrates how to fine-tune a **BERT model** for sentence-pair classification using the **Hugging Face Transformers library**. The example focuses on the dataset, which evaluates if two sentences are semantically equivalent.  

## Features  
- Load and preprocess the dataset using the `datasets` library.  
- Tokenize input data with `AutoTokenizer`.  
- Fine-tune a pretrained **BERT model** (`bert-base-uncased`) with a classification head for binary classification.  
- Use `DataCollatorWithPadding` for efficient batching and dynamic padding.  
- Implement a training loop with **PyTorch** and a learning rate scheduler.  
- Evaluate the fine-tuned model using evaluation metrics.  

## Requirements  
- Python 3.8+  
- PyTorch 1.10+  
- Hugging Face Transformers  
- Datasets library  
- TQDM  
- Evaluate  

Install the dependencies with:  
```bash
pip install torch transformers datasets tqdm evaluate
```
## How to Use  

### Clone the Repository  
```bash
git clone https://github.com/rajatbutola/LLM_finetuning_for_classification.git
```
## Run the Code
### Execute the script to fine-tune the model:

```bash
Copy code
python LLM_trainning.py
```
## Output
The code will display the training progress using a progress bar and output the evaluation results, including the accuracy and F1-score on the validation set.


## Evaluation Metrics
The model is evaluated using the specific metrics:

- Accuracy
- F1-score

## Example Dataset  

The dataset consists of sentence pairs labeled as **paraphrase** or **not paraphrase**:  

| Sentence 1            | Sentence 2             | Label            |  
|------------------------|------------------------|------------------|  
| "The sky is blue."     | "The sky appears blue." | 1 (Paraphrase)   |  
| "The car is red."      | "The book is open."     | 0 (Not Paraphrase) |  




## Model and Training Details
- Pretrained Model: bert-base-uncased
- Optimizer: AdamW (learning rate: 3e-5)
- Scheduler: Linear schedule with warmup
- Batch Size: 8
- Epochs: 3
## Results
Final evaluation metrics will be printed after training. You can expect a high F1-score on the dataset.

