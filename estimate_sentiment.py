import pandas as pd

# 1. Activate your virtual environment (if not already):
#    On Windows CMD:
#      venv\Scripts\activate.bat
#    On PowerShell:
#      .\venv\Scripts\Activate.ps1

from deeppavlov import build_model
import json

# 2. Point to your JSON config for the sentiment pipeline.
#    This should be the same config you used for training:
CONFIG_PATH = "./sentiment/sentiments_model_interfer.json"

# 3. Load the trained pipeline. 
#    Setting `download=False` skips re-downloading embeddings/etc.
sentiment_model = build_model(CONFIG_PATH, download=False)

df = pd.read_csv("./data/sentiment/test.csv", encoding="utf-8", header=0)


# Choose your chunk size:
chunk_size = 100

# Preallocate the three new columns with the correct length
df['estimated'] = None
df['pos_probe']     = None
df['neg_probe']     = None

# Iterate over idx ranges in steps of chunk_size
for start in range(0, len(df), chunk_size):
    end = min(start + chunk_size, len(df))
    sub_df = df.iloc[start:end]
    
    # Extract the texts for this chunk
    texts_chunk = sub_df['text'].tolist()
    
    # Call estimate() once on the chunk
    labels_chunk, rates_chunk = sentiment_model(texts_chunk)
    
    # Write labels back into df['estimated'][start:end]
    df.loc[start:end-1, 'estimated'] = labels_chunk
    
    # If rates_chunk is a list of [rate1, rate2], you can assign directly:
    df.loc[start:end-1, ['pos_probe', 'neg_probe']] = rates_chunk

df.to_csv('./data/results/test_bert.csv', index=False, encoding='utf-8')

