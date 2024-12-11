from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Suppress symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Sample data
texts = [
    "I love this product", "This is the worst experience ever", "Quite average but acceptable",
    "it is truly disgusting", "The script is unbelievable", "Swoozie Kurtz is excellent in a supporting role",
    "This movie is so wasteful of talent", "Robert DeNiro plays the most unbelievably intelligent illiterate of all time",
    "opening scene that is a terrific example of absurd comedy",
    "Story of a man who has unnatural feelings for a pig"
]
labels = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]

# Tokenize data
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

dataset = SentimentDataset(inputs, labels)

# Train-validation split using stratification
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Tokenize train and validation data
train_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
val_inputs = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

# Create train and validation datasets
train_dataset = SentimentDataset(train_inputs, train_labels)
val_dataset = SentimentDataset(val_inputs, val_labels)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# Optimizer and Loss
optimizer = AdamW(model.parameters(), lr=5e-5)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print(classification_report(true_labels, predictions, target_names=["Negative", "Positive"], zero_division=0))



"""   
#Faire des prédictions sur de nouveaux textes

# Nouveau texte
new_texts = ["The product is amazing!", "I didn't like the service."]
inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
inputs = {key: val.to(device) for key, val in inputs.items()}

# Prédiction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    sentiments = ["Negative" if pred == 0 else "Positive" for pred in preds]

print(sentiments)
"""
