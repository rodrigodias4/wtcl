import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

MAX_LENGTH = 512

# Device setup
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# Label mapping
label_list = ["O", "B", "I"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# Model definition
def build_model(hparams = None):
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    if hparams:
        model.config.hidden_dropout_prob = hparams["dropout"]
        model.config.attention_probs_dropout_prob = hparams["dropout"]
    model.to(get_device())
    return model

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def encode(text, labels):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    
    enc["labels"] = [label2id[label] for label in labels]
    return enc

# Dataset definition
class WTCLDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        enc = encode(item["text"], item["labels"])
        
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        
        # Set labels to -100 for padding tokens
        labels = torch.tensor(enc["labels"], dtype=torch.long)
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Evaluation metric
def compute_macro_f1_token_level(preds, labels, num_labels=3):

    preds = np.array(preds)
    labels = np.array(labels)

    # flatten (batch, seq) → (N,)
    preds = preds.flatten()
    labels = labels.flatten()

    # remove ignored tokens
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    f1s = []

    for c in range(num_labels):

        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        f1s.append(f1)

    return float(np.mean(f1s))

# ---------------------------------
# Training and Evaluation Functions
# ---------------------------------

def train(model, train_loader, optimizer, scheduler, device, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss / len(train_loader)}")

def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels

# -------------------------------
# Argument Parsing and Main Function
# -------------------------------

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Training for {args.num_epochs} epochs with batch size {args.batch_size} and learning rate {args.learning_rate}.")
    
    model = build_model()
    df = pd.read_csv(args.input_file)
    
    all_debates = df["debate"].unique()
    print(f"Found debates: {all_debates}")
    
    # Leave-one-debate-out cross-validation
    for debate in all_debates:
        print(f"Training model with debate '{debate}' left out for testing.")
        train_data = df[df["debate"] != debate]
        test_data = df[df["debate"] == debate]
        
        # Prepare datasets and dataloaders
        train_dataset = WTCLDataset(train_data.to_dict("records"), tokenizer, MAX_LENGTH)
        test_dataset = WTCLDataset(test_data.to_dict("records"), tokenizer, MAX_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

        total_steps = len(train_loader) * args.num_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Train the model
        train(model, train_loader, optimizer, scheduler, get_device(), args.num_epochs)
    

if __name__ == "__main__":
    main()