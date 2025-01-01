from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate

# Load the MRPC dataset from the GLUE benchmark
raw_datasets = load_dataset("glue", "mrpc")

# Define the model checkpoint to use (pretrained BERT model)
checkpoint = "bert-base-uncased"

# Load the tokenizer for the specified checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Function to tokenize the input sentences
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# Apply tokenization to the entire dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Data collator to handle dynamic padding during batching
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare data loaders for training and evaluation
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Peek into one batch to check the shapes of the data
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

# Load the pretrained BERT model with a classification head (2 output labels)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Define the optimizer with weight decay (AdamW)
optimizer = AdamW(model.parameters(), lr=3e-5)

# Move the model to GPU if available, otherwise to CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define the number of training epochs and steps
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# Scheduler for learning rate adjustment during training
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Progress bar to track training progress
progress_bar = tqdm(range(num_training_steps))

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Move the batch data to the appropriate device (CPU/GPU)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()

        # Update weights and scheduler step
        optimizer.step()
        lr_scheduler.step()
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Update the progress bar
        progress_bar.update(1)

# Load the evaluation metric (GLUE MRPC task-specific metric)
metric = evaluate.load("glue", "mrpc")

# Evaluation loop
model.eval()
for batch in eval_dataloader:
    # Move the batch data to the appropriate device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass without gradient computation (inference mode)
    with torch.no_grad():
        outputs = model(**batch)

    # Extract logits and compute predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    # Add predictions and references to the metric
    metric.add_batch(predictions=predictions, references=batch["labels"])

# Compute and print the final evaluation results
metric.compute()
