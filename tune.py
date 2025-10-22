import os
import pandas as pd
import mlflow
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load train/test
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

X_train = train_df.drop('left', axis=1)
y_train = train_df['left']
X_test = test_df.drop('left', axis=1)
y_test = test_df['left']

mlflow.set_experiment("HR Analytics Tuning")

# Parent run
with mlflow.start_run(run_name="SVC_tuning"):
    kernels = ["poly", "rbf"]
    Cs = [0.5, 1.0, 2.0]

    for kernel in kernels:
        for C in Cs:
            with mlflow.start_run(run_name=f"{kernel}_C{C}", nested=True):
                model = SVC(kernel=kernel, C=C, gamma="scale", random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                mlflow.log_param("kernel", kernel)
                mlflow.log_param("C", C)
                mlflow.log_metric("test_accuracy", acc)

                os.makedirs("models", exist_ok=True)
                model_path = f"models/model_{kernel}_C{C}.pkl"
                import joblib
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)

                print(f"Kernel={kernel}, C={C}, Test Accuracy={acc:.4f}")
