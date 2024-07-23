import pandas as pd
import numpy as np
import plotly.express as px
# import os
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch

ticket_df = pd.read_csv("./ticket-helpdesk-multi-lang.csv")

# Ensure all entries in text column are in string format and all lower case
ticket_df = ticket_df.dropna(subset=['text']) # drop null entries if existent
ticket_df['text'] = ticket_df['text'].str.lower()

# Encode Target Class queue
ticket_df['queue_enc'] = ticket_df['queue'].astype('category').cat.codes
# Get encoding mapping
q_map = dict(enumerate(ticket_df['queue'].astype('category').cat.categories))


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(ticket_df['text'], ticket_df['queue_enc'], test_size=0.2, random_state=42)

# Create Dataset object for HuggingFace
train_data = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
test_data = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

# Load pre-trained tokenizer and model
model_name = "xlm-roberta-base"  # Multilingual model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
        num_labels=ticket_df['queue_enc'].nunique(),
        id2label=q_map)


# Define  function to tokenize the text data from dataframe
def tokenizing_function(dataframe):
    return tokenizer(dataframe['text'], truncation=True, padding=True)

train_data = train_data.map(tokenizing_function, batched=True)
test_data = test_data.map(tokenizing_function, batched=True)

training_args = TrainingArguments(
    output_dir='./intelligence/results',
    eval_strategy="epoch",
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    use_mps_device=True
)

# Compute metrics for evaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': (preds == p.label_ids).astype(np.float32).mean().item(),
    }

# Instantiate the model trainer
model_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
print("Training Model : \n")
model_trainer.train()
print("\n")

# Evaluate the model trainer 
print("Model Evaluation: \n")
model_trainer.evaluate()
print("\n")
# Make Predictions With Model
predictions = model_trainer.predict(test_data)
preds = torch.tensor(predictions.predictions).argmax(axis=1)

# Show classification report
print("Model Classification Report: \n")
print(classification_report(y_test, preds, target_names=[q_map[i] for i in range(len(q_map))]))

print("\n")

# Show confusion matrix
cm = confusion_matrix(y_test, preds)
fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), x=list(q_map.values()), y=list(q_map.values()))
fig_cm.update_layout(title="Confusion Matrix")
fig_cm.show()

# Save the model
model_trainer.save_model("./intelligence/model")