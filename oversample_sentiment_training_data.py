import pandas as pd
import random
from tqdm import tqdm
from rephrase import rephrase

# 1. Load the original training CSV:
df = pd.read_csv('./data/sentiment/train.csv')

def augment_answer(answer_text: str) -> str:
    return rephrase(answer_text)

augmented_rows = []
N_SAMPLES = 1

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Oversampling NEG examples"): #df.iterrows():
    label = row['labels']
    full_text = row['text']
    
    # Split the text into "Q: ... " and "A: ..." parts
    # We assume there is exactly one "A:" in each row.
    if 'A:' in full_text:
        question_part, answer_part = full_text.split('A:', 1)
        answer_part = answer_part.strip()       # the actual answer text
        question_part = question_part.strip()   # the question part including "Q:"
    else:
        # If the format is unexpected, just skip augmentation for this row
        question_part, answer_part = full_text, ""
    
    # Always keep the original example
    augmented_rows.append(row)
    
    # If this example is NEG, create two augmented copies
    if label == 'NEG':
        for _ in range(N_SAMPLES):  # create N_SAMPLES more NEG variants
            new_answer = augment_answer(answer_part)
            new_text = f"{question_part} A: {new_answer}"
            
            new_row = row.copy()
            new_row['text'] = new_text
            augmented_rows.append(new_row)

# 2. Build a new DataFrame from the augmented rows
augmented_df = pd.DataFrame(augmented_rows)

# 3. Save to a new CSV
augmented_df.to_csv('./data/sentiment/train_oversampled.csv', index=False)

# 4. (Optional) Show new class distribution
print("Class counts after oversampling:")
print(augmented_df['labels'].value_counts())
