import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("🌸 SHAP Iris Classification Demo - Complete Hands-On Tutorial")
print("=" * 60)

# 1. Load the dataset
print("\n1. Loading Iris dataset...")
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print(f"   ✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# 2. Split the dataset into training and testing sets
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 3. Create and train a RandomForest model
print("\n3. Training RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
model.fit(X_train, y_train)
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"   ✅ Model trained | Train acc: {train_accuracy:.3f}, Test acc: {test_accuracy:.3f}")

# 4. Initialize the SHAP Explainer (TreeSHAP optimized for RandomForest)
print("\n4. Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
print("   ✅ TreeSHAP explainer ready (fast for tree models)")

# 5. Compute SHAP values for all test instances
print("\n5. Computing SHAP values...")
shap_values = explainer.shap_values(X_test)
print(f"   ✅ SHAP values computed: {np.array(shap_values).shape}")

# 6. SHAP Summary Plot (global feature importance)
print("\n6. Creating Summary Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Summary plot saved: shap_summary_plot.png")

# 7. FORCE PLOTS - EXERCISES SECTION
print("\n" + "="*60)
print("🎯 HANDS-ON EXERCISES - FORCE PLOTS")
print("="*60)

# Initialize JS for interactive plots (Jupyter/Colab)
shap.initjs()

# EXERCISE 1: Single Instance Force Plot (Instance 0)
print("\n📊 EXERCISE 1: Single Instance Force Plot")
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
shap.save_html("force_instance_0.html", 
               shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], show=False))
print("   ✅ Saved: force_instance_0.html (First test instance)")

# EXERCISE 2: Compare 3 Different Instances Side-by-Side
print("\n📊 EXERCISE 2: Compare Multiple Instances")
indices = [0, 5, 10]  # Try different indices here!
shap.force_plot(explainer.expected_value, 
                shap_values[indices], 
                X_test.iloc[indices])
shap.save_html("force_compare_3.html", 
               shap.force_plot(explainer.expected_value, shap_values[indices], X_test.iloc[indices], show=False))
print(f"   ✅ Saved: force_compare_3.html (Instances {indices})")

# EXERCISE 3: Stacked Force Plot (50 instances)
print("\n📊 EXERCISE 3: Stacked Force Plot (50 instances)")
shap.force_plot(explainer.expected_value, 
                shap_values[:50], 
                X_test.iloc[:50])
shap.save_html("force_stacked_50.html", 
               shap.force_plot(explainer.expected_value, shap_values[:50], X_test.iloc[:50], show=False))
print("   ✅ Saved: force_stacked_50.html")

# EXERCISE 4: Find and Explain a Misclassified Instance
print("\n📊 EXERCISE 4: Misclassified Instance Analysis")
y_pred = model.predict(X_test)
misclassified_idx = np.where(y_test != y_pred)[0]
if len(misclassified_idx) > 0:
    mis_idx = misclassified_idx[0]
    print(f"   🔍 Found misclassified instance at index {mis_idx}")
    print(f"   Actual: {y_test[mis_idx]}, Predicted: {y_pred[mis_idx]}")
    shap.force_plot(explainer.expected_value, shap_values[mis_idx], X_test.iloc[mis_idx])
    shap.save_html(f"force_misclassified_{mis_idx}.html", 
                   shap.force_plot(explainer.expected_value, shap_values[mis_idx], X_test.iloc[mis_idx], show=False))
    print(f"   ✅ Saved: force_misclassified_{mis_idx}.html")
else:
    print("   ✅ Perfect predictions! No misclassifications found.")

# EXERCISE 5: Class-Specific Force Plot (Multi-class example)
print("\n📊 EXERCISE 5: Class-Specific Force Plot (Setosa = class 0)")
shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0], 
                feature_names=X_test.columns.tolist(), show=False)
shap.save_html("force_class_setosa.html", 
               shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0], show=False))
print("   ✅ Saved: force_class_setosa.html (Setosa prediction forces)")

print("\n" + "="*60)
print("🎉 COMPLETE! All visualizations generated:")
print("📁 Files created:")
print("   • shap_summary_plot.png")
print("   • force_instance_0.html")
print("   • force_compare_3.html") 
print("   • force_stacked_50.html")
print("   • force_misclassified_X.html (if any)")
print("   • force_class_setosa.html")
print("\n🚀 Open HTML files in browser for interactive exploration!")
print("💡 Try changing indices [0,5,10] → [2,7,15] in Exercise 2!")
