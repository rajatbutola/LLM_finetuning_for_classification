# LLM_finetuning_for_classification
Fine-tuning a BERT model for sentence-pair classification on the GLUE MRPC dataset using Hugging Face Transformers.
# Fine-Tuning BERT on the GLUE MRPC Dataset  

This repository demonstrates how to fine-tune a **BERT model** for sentence-pair classification using the **Hugging Face Transformers library**. The example focuses on the **Microsoft Research Paraphrase Corpus (MRPC)** dataset from the **GLUE benchmark**, which evaluates if two sentences are semantically equivalent.  

## Features  
- Load and preprocess the MRPC dataset using the `datasets` library.  
- Tokenize input data with `AutoTokenizer`.  
- Fine-tune a pretrained **BERT model** (`bert-base-uncased`) with a classification head for binary classification.  
- Use `DataCollatorWithPadding` for efficient batching and dynamic padding.  
- Implement a training loop with **PyTorch** and a learning rate scheduler.  
- Evaluate the fine-tuned model using GLUE's MRPC evaluation metrics.  

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
git clone https://github.com/yourusername/bert-glue-mrpc.git
cd bert-glue-mrpc
```
## Run the Code
### Execute the script to fine-tune the model:

```bash
Copy code
python fine_tune_bert_mrpc.py
```
## Output
The code will display the training progress using a progress bar and output the evaluation results, including the accuracy and F1-score on the validation set.

## File Structure
- fine_tune_bert_mrpc.py: Main script for fine-tuning and evaluation.

## Evaluation Metrics
The model is evaluated using the GLUE MRPC-specific metrics:

- Accuracy
- F1-score

## Example Dataset  

The MRPC dataset consists of sentence pairs labeled as **paraphrase** or **not paraphrase**:  

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
Final evaluation metrics will be printed after training. You can expect a high F1-score on the MRPC dataset.

