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
