import os
import joblib
import pandas as pd
import mlflow
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load training data
train_df = pd.read_csv('data/train.csv')
X_train = train_df.drop('left', axis=1)
y_train = train_df['left']

# Train model
model = SVC(kernel='poly', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("SVC model saved to models/model.pkl")

# Compute metrics
train_acc = accuracy_score(y_train, model.predict(X_train))

# MLflow logging
mlflow.set_experiment("HR Analytics Baseline")

with mlflow.start_run(run_name="SVC_poly_baseline"):
    mlflow.log_param("kernel", "poly")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("gamma", "scale")
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_artifact("models/model.pkl")

print(f"Training accuracy: {train_acc:.4f}")
