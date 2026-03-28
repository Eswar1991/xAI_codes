import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import numpy as np
import os

print("🎬 LIME IMDB Sentiment Demo - Complete Hands-On Tutorial")
print("=" * 70)

# Check if dataset exists
if not os.path.exists("aclImdb/train"):
    print("❌ ERROR: Download IMDB dataset first!")
    print("   wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
    print("   tar -xzf aclImdb_v1.tar.gz")
    exit(1)

# 1. Load the IMDB dataset
print("\n1. Loading IMDB dataset...")
data = load_files("aclImdb/train", categories=["pos", "neg"], 
                  encoding="utf-8", decode_error="replace")
X = np.array(data.data)
y = data.target
print(f"   ✅ Loaded: {len(X):,} reviews | Positive: {sum(y==1):,}, Negative: {sum(y==0):,}")

# 2. Split the dataset
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✅ Train: {len(X_train):,}, Test: {len(X_test):,}")

# 3. Create the text classification pipeline
print("\n3. Creating TF-IDF + LogisticRegression pipeline...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
classifier = LogisticRegression(random_state=42, max_iter=1000)
pipeline = make_pipeline(vectorizer, classifier)

# 4. Train the model
print("\n4. Training model...")
pipeline.fit(X_train, y_train)
train_acc = pipeline.score(X_train, y_train)
test_acc = pipeline.score(X_test, y_test)
print(f"   ✅ Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")

# 5. Initialize LIME explainer
print("\n5. Initializing LIME explainer...")
explainer = lime.lime_text.LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])

print("\n" + "="*70)
print("🎯 HANDS-ON EXERCISES")
print("="*70)

# EXERCISE 1: Baseline explanation (num_features=10)
print("\n📊 EXERCISE 1: Baseline LIME (10 features)")
text_instance = X_test[0]
print(f"   Analyzing: '{text_instance[:100]}...'")
exp1 = explainer.explain_instance(
    text_instance, pipeline.predict_proba, num_features=10
)
exp1.save_to_file('exercise1_baseline_10.html')
print("   ✅ Saved: exercise1_baseline_10.html")

# EXERCISE 2: Fewer features (num_features=5) - Cleaner explanation
print("\n📊 EXERCISE 2: Fewer Features (num_features=5)")
exp2 = explainer.explain_instance(
    text_instance, pipeline.predict_proba, num_features=5
)
exp2.save_to_file('exercise2_fewer_features_5.html')
print("   ✅ Saved: exercise2_fewer_features_5.html")

# EXERCISE 3: More features (num_features=20) - Detailed explanation
print("\n📊 EXERCISE 3: More Features (num_features=20)")
exp3 = explainer.explain_instance(
    text_instance, pipeline.predict_proba, num_features=20
)
exp3.save_to_file('exercise3_more_features_20.html')
print("   ✅ Saved: exercise3_more_features_20.html")

# EXERCISE 4: Different TF-IDF parameters (max_features=1000)
print("\n📊 EXERCISE 4: Limited Vocabulary (max_features=1000)")
vectorizer4 = TfidfVectorizer(max_features=1000, ngram_range=(1,1))
classifier4 = LogisticRegression(random_state=42, max_iter=1000)
pipeline4 = make_pipeline(vectorizer4, classifier4)
pipeline4.fit(X_train, y_train)
exp4 = explainer.explain_instance(
    text_instance, pipeline4.predict_proba, num_features=10
)
exp4.save_to_file('exercise4_small_vocab.html')
print("   ✅ Saved: exercise4_small_vocab.html")

# EXERCISE 5: N-grams (word pairs) for context
print("\n📊 EXERCISE 5: N-grams (1,2) - Captures phrases")
vectorizer5 = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
classifier5 = LogisticRegression(random_state=42, max_iter=1000)
pipeline5 = make_pipeline(vectorizer5, classifier5)
pipeline5.fit(X_train, y_train)
exp5 = explainer.explain_instance(
    text_instance, pipeline5.predict_proba, num_features=10
)
exp5.save_to_file('exercise5_ngrams.html')
print("   ✅ Saved: exercise5_ngrams.html")

# EXERCISE 6: Different test instance (negative review)
print("\n📊 EXERCISE 6: Different Test Instance")
neg_idx = np.where(y_test == 0)[0][0]  # First negative review
text_neg = X_test[neg_idx]
print(f"   Analyzing negative review: '{text_neg[:100]}...'")
exp6 = explainer.explain_instance(
    text_neg, pipeline.predict_proba, num_features=10
)
exp6.save_to_file('exercise6_negative_review.html')
print("   ✅ Saved: exercise6_negative_review.html")

# EXERCISE 7: Low convergence (max_iter=200)
print("\n📊 EXERCISE 7: Poor Convergence (max_iter=200)")
vectorizer7 = TfidfVectorizer(max_features=5000)
classifier7 = LogisticRegression(random_state=42, max_iter=200)
pipeline7 = make_pipeline(vectorizer7, classifier7)
pipeline7.fit(X_train, y_train)
print(f"   Accuracy with poor convergence: {pipeline7.score(X_test, y_test):.3f}")
exp7 = explainer.explain_instance(
    text_instance, pipeline7.predict_proba, num_features=10
)
exp7.save_to_file('exercise7_poor_convergence.html')
print("   ✅ Saved: exercise7_poor_convergence.html")

print("\n" + "="*70)
print("🎉 COMPLETE! Generated 7 HTML explanations:")
print("📁 Files created:")
print("   • exercise1_baseline_10.html          (Original 10 features)")
print("   • exercise2_fewer_features_5.html     (Cleaner: 5 features)")
print("   • exercise3_more_features_20.html     (Detailed: 20 features)")
print("   • exercise4_small_vocab.html          (Limited vocab: 1000 words)")
print("   • exercise5_ngrams.html              (Word pairs context)")
print("   • exercise6_negative_review.html      (Negative prediction)")
print("   • exercise7_poor_convergence.html     (Poor model quality)")
print("\n🚀 Open HTML files in browser to compare explanations!")
print("\n💡 Quick tests - Change these parameters in the script:")
print("   • num_features: [5, 10, 15, 20]")
print("   • max_features: [1000, 5000, None]")
print("   • ngram_range: (1,1), (1,2), (1,3)")
print("   • max_iter: [200, 500, 1000]")
