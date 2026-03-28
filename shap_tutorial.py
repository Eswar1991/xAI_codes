import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load the dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Initialize the SHAP Explainer
# We use model.predict_proba or model.predict depending on the goal
explainer = shap.Explainer(model.predict, X_test)

# 5. Compute SHAP values
shap_values = explainer(X_test)

# Plot SHAP Summary Plot
shap.summary_plot(shap_values, X_test, show=False) # Tell SHAP not to auto-close
plt.show() # Force the window to pop up on your Mac