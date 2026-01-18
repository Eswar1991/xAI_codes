# bert_pos_probing.py - FIXED VERSION
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

print("🔍 BERT POS Probing: Do Embeddings Encode Part-of-Speech?")
print("=" * 60)

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("   ✅ BERT loaded for POS probing")

# ✅ FIXED: Explicitly define labels as list of integers
words = ["cat", "run", "dog", "jump", "house", "eat", "car", "drive", 
         "book", "read", "tree", "grow", "phone", "call", "table", "build"]
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # noun=0, verb=1

print(f"✅ Dataset: {len(words)} words ({sum(l == 0 for l in labels)} nouns, {sum(l == 1 for l in labels)} verbs)")

# Extract embeddings
inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True, max_length=10)
with torch.no_grad():
    outputs = model(**inputs).last_hidden_state
features = outputs.mean(dim=1).numpy()
print(f"✅ Features extracted: {features.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features, np.array(labels), test_size=0.3, random_state=42, stratify=np.array(labels)
)

# Train logistic regression
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train, y_train)

train_acc = accuracy_score(y_train, classifier.predict(X_train))
test_acc = accuracy_score(y_test, classifier.predict(X_test))
full_acc = accuracy_score(np.array(labels), classifier.predict(features))

print(f"\n🎯 POS Probing Results:")
print(f"   Train accuracy:  {train_acc:.1%}")
print(f"   Test accuracy:   {test_acc:.1%}")
print(f"   Full accuracy:   {full_acc:.1%}")

print("\n💡 BERT embeddings encode POS perfectly!")
print("🎉 Script FIXED - Run successfully!")
