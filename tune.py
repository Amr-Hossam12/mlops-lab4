import os
import joblib
import pandas as pd
import mlflow
from dotenv import load_dotenv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

load_dotenv()

# Load train/test data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

X_train = train_df.drop('left', axis=1)
y_train = train_df['left']
X_test = test_df.drop('left', axis=1)
y_test = test_df['left']

# Set up MLflow experiment
mlflow.set_experiment("HR Analytics Tuning")

best_acc = -1.0
best_model = None
best_params = {}

# Parent run
with mlflow.start_run(run_name="SVC_tuning"):
    kernels = ["poly", "rbf"]
    Cs = [0.5, 1.0, 2.0]

    for kernel in kernels:
        for C in Cs:
            with mlflow.start_run(run_name=f"{kernel}_C{C}", nested=True):
                # Train model
                model = SVC(kernel=kernel, C=C, gamma="scale", random_state=42)
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # Log to MLflow
                mlflow.log_param("kernel", kernel)
                mlflow.log_param("C", C)
                mlflow.log_metric("test_accuracy", acc)

                print(f"Kernel={kernel}, C={C}, Test Accuracy={acc:.4f}")

                # Track best model
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = {"kernel": kernel, "C": C}

    # Save and log the best model
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/best_svc.pkl"
    joblib.dump(best_model, best_model_path)
    mlflow.log_artifact(best_model_path)

    mlflow.log_metric("best_test_accuracy", best_acc)
    mlflow.log_param("best_params", str(best_params))

print(f"Best Model: {best_params} → Accuracy: {best_acc:.4f}")
print("Saved best model to models/best_svc.pkl")
