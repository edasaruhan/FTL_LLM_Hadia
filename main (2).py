# i. Install the Necessary Libraries and Tools
!pip install transformers datasets matplotlib seaborn

# ii. Exploratory Data Analysis (EDA)
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (for example, an essay scoring dataset)
dataset = load_dataset('path/to/dataset')

# Display basic information about the dataset
print(dataset)

# Visualize the dataset (e.g., distribution of scores)
sns.histplot(dataset['train']['score'])
plt.title('Distribution of Essay Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# iii. Dataset Preparation
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the essays
def tokenize_function(examples):
    return tokenizer(examples['essay'], truncation=True, padding='max_length')

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# iv. Model Selection
from transformers import AutoModelForSequenceClassification

# Load a pre-trained model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# v. Finetuning Process
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
)

trainer.train()

# vi. Evaluation
from sklearn.metrics import mean_squared_error

# Evaluate the model
predictions = trainer.predict(tokenized_dataset['test'])
predicted_scores = predictions.predictions.flatten()
true_scores = tokenized_dataset['test']['score']

# Calculate and print the mean squared error
mse = mean_squared_error(true_scores, predicted_scores)
print(f'Mean Squared Error: {mse}')

# Compare performance before and after fine-tuning (for example, using validation set)
# (Assuming you have initial model predictions before fine-tuning)
initial_predictions = ...  # Load or compute initial predictions
initial_mse = mean_squared_error(true_scores, initial_predictions)
print(f'Initial Mean Squared Error: {initial_mse}')
print(f'Improvement: {initial_mse - mse}')
