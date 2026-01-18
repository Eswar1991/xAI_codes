# bert_pca_embeddings.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel

print("🤖 BERT Word Embeddings PCA Visualization")
print("=" * 50)

# Load PyTorch BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("   ✅ BERT-base-uncased (PyTorch) loaded")

# Semantic word pairs for analysis
words = ["king", "queen", "man", "woman", "Paris", "France", "Berlin", "Germany"]
print(f"\n2. Analyzing {len(words)} words: {words}")

# Tokenize and extract embeddings
inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs).last_hidden_state  # [batch, seq_len, 768]
features = outputs.mean(dim=1).numpy()  # Pool across tokens: [8, 768]
print(f"   ✅ Embeddings shape: {features.shape}")

# PCA to 2D visualization
pca = PCA(n_components=2)
embed_2d = pca.fit_transform(features)
print(f"   📊 Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embed_2d[:, 0], embed_2d[:, 1], s=300, c='royalblue', alpha=0.7, edgecolors='black', linewidth=2)
for i, word in enumerate(words):
    plt.annotate(word, (embed_2d[i, 0], embed_2d[i, 1]), xytext=(8, 8), textcoords='offset points',
                fontsize=16, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
plt.title("BERT Word Embeddings: Semantic Clusters (PCA 2D)", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Principal Component 1", fontsize=14)
plt.ylabel("Principal Component 2", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bert_pca_embeddings.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n🎯 Expected: king/man cluster, queen/woman cluster, city/country pairs!")
print("📁 Saved: bert_pca_embeddings.png")
