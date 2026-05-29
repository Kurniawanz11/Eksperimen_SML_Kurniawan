import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import dagshub

dagshub.init(
    repo_owner="Kurniawanz11",
    repo_name="heart-disease-mlops",
    mlflow=True
)


# Tracking URI
# mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_tracking_uri("file:mlruns")
# Experiment
mlflow.set_experiment("Heart_Disease_Tuning")
mlflow.autolog()

# Load data
X_train = pd.read_csv("preprocessing/heart_preprocessing/X_train.csv")
X_test = pd.read_csv("preprocessing/heart_preprocessing/X_test.csv")

y_train = pd.read_csv("preprocessing/heart_preprocessing/y_train.csv")
y_test = pd.read_csv("preprocessing/heart_preprocessing/y_test.csv")


with mlflow.start_run():

    # Parameter tuning
    param_grid = {

        "n_estimators": [50, 100],

        "max_depth": [5, 10],

        "min_samples_split": [2, 5]
    }

    # Model
    rf = RandomForestClassifier(
        random_state=42
    )

    # Grid Search
    grid_search = GridSearchCV(

        estimator=rf,

        param_grid=param_grid,

        cv=3,

        scoring="accuracy",

        n_jobs=-1
    )

    # Training
    grid_search.fit(
        X_train,
        y_train.values.ravel()
    )

    # Best model
    best_model = grid_search.best_estimator_

    # Prediction
    y_pred = best_model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Best params
    best_params = grid_search.best_params_

    print("Best Parameter:")
    print(best_params)

    print("Accuracy:")
    print(accuracy)

    # Logging parameter
    mlflow.log_params(best_params)

    # Logging metric
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # Save model
    joblib.dump(
        best_model,
        "Membangun_model/best_random_forest.pkl"
    )

    # Log artifact
    mlflow.log_artifact(
        "Membangun_model/best_random_forest.pkl"
    )

    # Classification report
    report = classification_report(
        y_test,
        y_pred
    )

    with open(
        "Membangun_model/classification_report_tuning.txt",
        "w"
    ) as f:

        f.write(report)

    mlflow.log_artifact(
        "Membangun_model/classification_report_tuning.txt"
    )

    # Log model
    mlflow.sklearn.log_model(

        sk_model=best_model,

        artifact_path="best_model",

        input_example=X_train.iloc[:5]
    )

    print("Hyperparameter tuning selesai")