# bert_layer_activations.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

print("🧠 BERT Layer-wise Activation Profiling")
print("=" * 50)

# Load BERT with hidden states
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("   ✅ BERT loaded with 13 hidden states (embeddings + 12 layers)")

def extract_layer_activations(model, sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # List of 13 tensors
        layer_means = [torch.mean(state).cpu().numpy() for state in hidden_states]
    return np.array(layer_means)

# Test 3 diverse sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning revolutionized artificial intelligence.",
    "BERT captures contextual word relationships perfectly."
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

for i, sentence in enumerate(sentences):
    means = extract_layer_activations(model, sentence)
    axes[i].plot(range(13), means, marker='o', color=colors[i], linewidth=3, markersize=8)
    axes[i].set_title(f"'{sentence[:35]}...'", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Layer (0=Embeddings, 12=Final)", fontsize=11)
    axes[i].set_ylabel("Mean Activation", fontsize=11)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xticks(range(0, 13, 2))

plt.suptitle("BERT Layer Activations: Embeddings → Contextual Understanding", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("bert_layer_activations.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 Layers breakdown:")
print("   0: Raw embeddings | 1-6: Syntax | 7-11: Semantics | 12: Context")
print("📁 Saved: bert_layer_activations.png")
