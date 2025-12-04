# train_model.py
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load Data
print("Loading Iris dataset...")
data = load_iris(as_frame=True)
X = data.data
y = data.target

# 2. Simple ML Pipeline Definition
# We use a pipeline to ensure any necessary preprocessing (like scaling) is also saved.
print("Defining ML pipeline...")
ml_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 3. Train the Model
# We train on the full dataset here for simplicity in a demo, 
# but in production, you would use a dedicated training set.
print("Training model...")
ml_pipeline.fit(X, y)

# 4. Save Model Artifact and Class Names
# Save the entire pipeline, not just the classifier.
MODEL_PATH = 'iris_model.joblib'
CLASS_NAMES = list(data.target_names)

joblib.dump(ml_pipeline, MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")
print(f"Class names: {CLASS_NAMES}")
