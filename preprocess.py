# preprocess.py
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Load label mapping
with open('./data/training/label2id.json', 'r', encoding='utf-8') as f:
    label2id = json.load(f)

# Process dataset
df = pd.read_csv('./data/training/data.csv')
df['label_id'] = df['label'].map(label2id)

dev_ratio: float = 0.15
test_ratio: float = 0.15

# Split data (80% train, 20% validation)
train_df, temp_df = train_test_split(
    df[['text', 'label_id']], 
    test_size=dev_ratio + test_ratio,
    random_state=42
)

rel_dev = dev_ratio / (dev_ratio + test_ratio)
valid_df, test_df = train_test_split(
    temp_df,
    test_size=1 - rel_dev,
    random_state=42
)

# Save datasets
train_df.to_csv('./data/training/train.txt', sep='\t', index=False, header=False)
valid_df.to_csv('./data/training/valid.txt', sep='\t', index=False, header=False)
valid_df.to_csv('./data/training/test.txt', sep='\t', index=False, header=False)

# Create label vocabulary
id2label = {v: k for k, v in label2id.items()}
with open('./data/training/label_names.txt', 'w', encoding='utf-8') as f:
    for i in range(len(label2id)):
        f.write(id2label[i] + '\n')