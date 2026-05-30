import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import os
import dagshub

dagshub.init(
    repo_owner="Kurniawanz11",
    repo_name="heart-disease-mlops",
    mlflow=True
)
# ==========================================
# Load Dataset
# ==========================================

X_train = pd.read_csv(
    "preprocessing/heart_preprocessing/X_train.csv"
)

X_test = pd.read_csv(
    "preprocessing/heart_preprocessing/X_test.csv"
)

y_train = pd.read_csv(
    "preprocessing/heart_preprocessing/y_train.csv"
)

y_test = pd.read_csv(
    "preprocessing/heart_preprocessing/y_test.csv"
)

# ==========================================
# MLflow Experiment
# ==========================================

#mlflow.set_tracking_uri("file:mlruns")
mlflow.set_experiment("Heart_Disease_Tuning")

# ==========================================
# Hyperparameter Tuning
# ==========================================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5]
}

with mlflow.start_run():

    model = RandomForestClassifier(
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    # Training
    grid_search.fit(
        X_train,
        y_train.values.ravel()
    )

    # Best Model
    best_model = grid_search.best_estimator_

    # Prediction
    y_pred = best_model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # ==========================================
    # MANUAL LOGGING PARAMETER
    # ==========================================

    mlflow.log_param(
        "n_estimators",
        grid_search.best_params_["n_estimators"]
    )

    mlflow.log_param(
        "max_depth",
        grid_search.best_params_["max_depth"]
    )

    mlflow.log_param(
        "min_samples_split",
        grid_search.best_params_["min_samples_split"]
    )

    # ==========================================
    # MANUAL LOGGING METRIC
    # ==========================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # ==========================================
    # SAVE ARTIFACT
    # ==========================================

    os.makedirs(
        "artifacts_tuning",
        exist_ok=True
    )

    # Save model
    model_path = (
        "artifacts_tuning/"
        "best_random_forest.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    mlflow.log_artifact(
        model_path
    )

    # ==========================================
    # Classification Report
    # ==========================================

    report = classification_report(
        y_test,
        y_pred
    )

    report_path = (
        "artifacts_tuning/"
        "classification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(report)

    mlflow.log_artifact(
        report_path
    )

    # ==========================================
    # Confusion Matrix
    # ==========================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    cm_path = (
        "artifacts_tuning/"
        "confusion_matrix.txt"
    )

    with open(
        cm_path,
        "w"
    ) as f:

        f.write(
            str(cm)
        )

    mlflow.log_artifact(
        cm_path
    )

    # ==========================================
    # Best Parameter
    # ==========================================

    parameter_path = (
        "artifacts_tuning/"
        "best_parameter.txt"
    )

    with open(
        parameter_path,
        "w"
    ) as f:

        f.write(
            str(
                grid_search.best_params_
            )
        )

    mlflow.log_artifact(
        parameter_path
    )

    # ==========================================
    # Log Model
    # ==========================================

    mlflow.sklearn.log_model(
        best_model,
        "best_model"
    )

    # ==========================================
    # Output
    # ==========================================

    print(
        "Best Parameter :",
        grid_search.best_params_
    )

    print(
        "Accuracy :",
        accuracy
    )