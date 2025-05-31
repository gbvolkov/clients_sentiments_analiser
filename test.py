import json
from deeppavlov import build_model

# ─── Step 1: Build your DeepPavlov model ───────────────────────────────────────
CONFIG_PATH = "./sentiment/sentiments_model.json"
model = build_model(CONFIG_PATH, download=False)

# ─── Step 2: Manually load and invert classes.dict ─────────────────────────────
# Note: Adjust this path if your classes.dict is elsewhere.
CLASSES_PATH = "C:/Users/volko/.deeppavlov/models/classifiers/sentiment_twitter_torch/classes.dict"
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    token2id = json.load(f)
# token2id might be {"NEG": 0, "POS": 1} or similar.
id2token = {idx: tok for tok, idx in token2id.items()}

# ─── Step 3: Prepare some test sentences ───────────────────────────────────────
texts = [
    "Всё ужасно!",                     # clearly negative
    "Мне очень понравилось!",           # clearly positive
    "Нормально, но могло быть лучше."   # likely neutral/ambiguous
]

# ─── Step 4: Run inference to get (labels, probabilities) ────────────────────
# Your JSON must have:
#   "out": ["y_pred_labels", "y_pred_probas"]
# in the chainer, and the torch_text_classification_model must output "probas" in its "out" list.
labels, probs = model(texts)

# ─── Step 5: Print default outputs ─────────────────────────────────────────────
print("=== Default DeepPavlov outputs ===\n")
for t, lbl, score_vec in zip(texts, labels, probs):
    print(f"Text:    {t!r}")
    print(f"Default Label: {lbl}")
    print(f"Scores:        {score_vec}")  # e.g. [p_NEG, p_POS] or [p_POS, p_NEG], depending on your classes order
    print("-" * 40)

# ─── Step 6: Override using a simple threshold (p_NEG > 0.5 → NEG, else POS) ─
print("\n=== Threshold-based override (p_NEG > 0.5 → NEG, else POS) ===\n")
for t, score_vec in zip(texts, probs):
    # Determine index order (example assumes token2id = {"NEG": 0, "POS": 1})
    # If your token2id is reversed, swap these two lines accordingly.
    neg_idx = token2id["NEG"]
    pos_idx = token2id["POS"]

    p_neg = score_vec[neg_idx]
    p_pos = score_vec[pos_idx]

    # Choose "NEG" if its probability > 0.5, else "POS"
    if p_neg > 0.5:
        override_idx = neg_idx
    else:
        override_idx = pos_idx

    override_label = id2token[override_idx]
    print(f"Text:   {t!r}")
    print(f"→ p_NEG = {p_neg:.3f}, p_POS = {p_pos:.3f}")
    print(f"→ Override Label: {override_label}")
    print("-" * 40)
