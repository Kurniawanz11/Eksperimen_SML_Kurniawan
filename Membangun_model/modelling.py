import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import dagshub


dagshub.init(
    repo_owner="Kurniawanz11",
    repo_name="heart-disease-mlops",
    mlflow=True
)

#mlflow.set_tracking_uri("file:mlruns")

# Load data
X_train = pd.read_csv("preprocessing/heart_preprocessing/X_train.csv")
X_test = pd.read_csv("preprocessing/heart_preprocessing/X_test.csv")

y_train = pd.read_csv("preprocessing/heart_preprocessing/y_train.csv")
y_test = pd.read_csv("preprocessing/heart_preprocessing/y_test.csv")


# MLflow experiment
mlflow.set_experiment("Heart_Disease_Experiment")
mlflow.autolog()

with mlflow.start_run():

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train.values.ravel())

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Logging parameter
    mlflow.log_param("n_estimators", 100)

    # Logging metric
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    joblib.dump(model, "Membangun_model/random_forest_model.pkl")

    # Logging artifact
    mlflow.log_artifact("Membangun_model/random_forest_model.pkl")

    # Classification report
    report = classification_report(y_test, y_pred)

    with open("Membangun_model/classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("Membangun_model/classification_report.txt")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    with open("Membangun_model/confusion_matrix.txt", "w") as f:
        f.write(str(cm))

    mlflow.log_artifact("Membangun_model/confusion_matrix.txt")

    # Log model
    mlflow.sklearn.log_model(
        model,
        "Membangun_model/random_forest_model"
    )

    print("Training selesai")
    print("Accuracy:", accuracy)