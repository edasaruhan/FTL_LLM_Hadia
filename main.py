from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Part 1: Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Part 2: Data Preparation and Testing
# Sample input
text = "I love using Hugging Face!"

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt")

# Get model predictions
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# Print prediction
print(f"Predicted label: {predictions.item()}")

# Part 2: Fine-Tuning the Model
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Replace with your actual train dataset
    eval_dataset=eval_dataset,    # Replace with your actual eval dataset
)

# Fine-tune the model
trainer.train()

# Part 3: Evaluation
# Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation results
print(f"Evaluation results: {eval_results}")
