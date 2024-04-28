import os
import pandas as pd
import json
from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    BitsAndBytesConfig
)

# Path to the downloaded CSV file
csv_file_path = 'leet.csv'

# Load the dataset from CSV
df = pd.read_csv(csv_file_path)
# Convert the DataFrame to a Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Define the model and tokenizer
model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Preprocess the dataset for model input
def preprocess_function(examples):
    return tokenizer(examples['text_column_name'], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    report_to="none"  # Change to "all" to view logs during training
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./trained_model')

# Select the first 100 entries for prediction
subset_dataset = tokenized_datasets.select(range(100))

# Generate predictions for the selected subset of the dataset
def generate_predictions(test_samples):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    for sample in test_samples:
        inputs = tokenizer(sample['text_column_name'], return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model.generate(**inputs)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append({'question': sample['text_column_name'], 'answer': decoded_output})
    return predictions

# Generate predictions
model_predictions = generate_predictions(subset_dataset)

# Save predictions to a JSON file
output_file_path = './model_predictions.json'
with open(output_file_path, 'w') as f:
    json.dump(model_predictions, f)

print(f"Predictions saved to {output_file_path}")
