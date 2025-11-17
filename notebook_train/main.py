# ------------------------------------------
# 1. ENVIRONMENT SETUP (GPU FORCE)
# ------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # Force GPU use

import torch

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠ No GPU detected — check Kaggle settings")

# ------------------------------------------
# 2. INSTALL CORE LIBRARIES (SAFE MODE)
# ------------------------------------------
!pip install transformers datasets evaluate --quiet --no-deps

# ------------------------------------------
# 3. IMPORTS
# ------------------------------------------
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ------------------------------------------
# 4. LOAD DATASET
# ------------------------------------------
dataset = load_dataset("imdb")

# ------------------------------------------
# 5. TOKENIZATION
# ------------------------------------------
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

tokenized_ds = dataset.map(tokenize, batched=True)
tokenized_ds = tokenized_ds.remove_columns(["text"])
tokenized_ds = tokenized_ds.rename_column("label", "labels")

# ------------------------------------------
# 6. LOAD MODEL (FORCE GPU)
# ------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Ensure model uses GPU
if torch.cuda.is_available():
    model.to("cuda")

# ------------------------------------------
# 7. EVALUATION METRICS
# ------------------------------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    return accuracy.compute(predictions=preds, references=labels)

# ------------------------------------------
# 8. TRAINING ARGUMENTS (OPTIMAL)
# ------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    run_name="distilbert_sentiment_run",
    report_to="none",                   # Disable wandb
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,                          # 🔥 Rocket speed on GPU
    gradient_accumulation_steps=1,
    logging_steps=50,
)

# ------------------------------------------
# 9. TRAINER
# ------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metrics
)

# ------------------------------------------
# 10. TRAIN THE MODEL
# ------------------------------------------
trainer.train()

# ------------------------------------------
# 11. SAVE MODEL
# ------------------------------------------
trainer.save_model("./results")
print("🎉 Training completed successfully!")

import shutil

shutil.make_archive("distilbert_model_output", 'zip', "./results")

