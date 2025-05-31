import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

# Configuration
MAX_LEN       = 256
BATCH_SIZE    = 32
EPOCHS        = 8
LEARNING_RATE = 2e-5
MODEL_NAME    = "DeepPavlov/rubert-base-cased"
SAVE_PATH     = "sentiment_analysis/model"
NUM_LABELS    = 38
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1  # fraction of total steps

# Dataset definition
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label':          torch.tensor(label, dtype=torch.long)
        }

# Helper to load CSVs
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].astype(int)
    return df['text'].tolist(), df['label'].tolist()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Load train/validation splits
train_texts, train_labels = load_data("./data/training/train.csv")
valid_texts, valid_labels = load_data("./data/training/valid.csv")

# 1) Compute class weights for the loss
classes = np.arange(NUM_LABELS)
weights = compute_class_weight(class_weight="balanced",
                               classes=classes,
                               y=train_labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# 2) Option A: WeightedRandomSampler to balance batches
#    (alternate: omit sampler and rely on Loss weighting alone)
sample_weights = [1.0 / Counter(train_labels)[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Prepare datasets & loaders
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
valid_dataset = TextDataset(valid_texts, valid_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Model, optimizer, scheduler, and loss
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
).to(device)

optimizer = AdamW(model.parameters(),
                  lr=LEARNING_RATE,
                  weight_decay=WEIGHT_DECAY)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps
)

criterion = CrossEntropyLoss(weight=class_weights)

# Training & validation loop
for epoch in range(1, EPOCHS + 1):
    # — Training —
    model.train()
    train_bar = tqdm(train_loader,
                     desc=f"Epoch {epoch}/{EPOCHS} [Train]",
                     leave=False,
                     miniters=1)
    total_train_loss = 0.0

    for batch in train_bar:
        optimizer.zero_grad()
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
        logits  = outputs.logits
        loss    = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        train_bar.set_postfix(train_loss=f"{loss.item():.4f}")
        train_bar.update(1)

    avg_train_loss = total_train_loss / len(train_loader)

    # — Validation —
    model.eval()
    val_bar = tqdm(valid_loader,
                   desc=f"Epoch {epoch}/{EPOCHS} [Valid]",
                   leave=False,
                   miniters=1)
    val_preds, val_true = [], []

    with torch.no_grad():
        for batch in val_bar:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
            logits  = outputs.logits
            preds   = torch.argmax(logits, dim=1).cpu().numpy()

            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())

            batch_acc = (preds == labels.cpu().numpy()).mean()
            val_bar.set_postfix(batch_acc=f"{batch_acc:.3f}")
            val_bar.update(1)

    accuracy = accuracy_score(val_true, val_preds)
    f1       = f1_score(val_true, val_preds, average='macro')

    print(f"Epoch {epoch}/{EPOCHS} — "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Acc: {accuracy:.4f} | "
          f"Val F1 (macro): {f1:.4f}")

# Save model & tokenizer
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model and tokenizer saved to {SAVE_PATH}")
