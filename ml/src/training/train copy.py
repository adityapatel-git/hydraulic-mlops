from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("ml/data/processed/hydraulic_features.csv")
MODEL_DIR = Path("ml/models")

TARGET = "valve_condition"

LABELS = [73, 80, 90, 100]


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Only use engineered sensor features.
    # Do NOT use the other condition labels because
    # they would cause target leakage.
    excluded_columns = [
        "cooler_condition",
        "valve_condition",
        "pump_leakage",
        "accumulator_pressure",
        "stable_flag",
    ]

    X = df.drop(columns=excluded_columns)
    y = df[TARGET]

    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # Stratified split keeps class proportions similar
    # between training and testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # --------------------------------------------------
    # MLflow
    # --------------------------------------------------

    mlflow.set_experiment("hydraulic-valve-condition")

    with mlflow.start_run():

        # --------------------------------------------------
        # Model configuration
        # --------------------------------------------------

        n_estimators = 300
        class_weight = "balanced"
        random_state = 42

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_samples", X.shape[0])

        # --------------------------------------------------
        # Train model
        # --------------------------------------------------

        print("\nTraining Random Forest...")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        print("Training complete.")

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        y_pred = model.predict(X_test)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        weighted_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
        )

        accuracy = (y_test == y_pred).mean()

        # Log metrics to MLflow
        mlflow.log_metric("weighted_f1", weighted_f1)
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("accuracy", accuracy)

        # --------------------------------------------------
        # Console output
        # --------------------------------------------------

        print("\n===== MODEL PERFORMANCE =====")

        print(f"\nWeighted F1: {weighted_f1:.4f}")
        print(f"Macro F1:    {macro_f1:.4f}")
        print(f"Accuracy:    {accuracy:.4f}")

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                y_pred,
                labels=LABELS,
            )
        )

        print("\nConfusion Matrix:")

        print(
            confusion_matrix(
                y_test,
                y_pred,
                labels=LABELS,
            )
        )

        # --------------------------------------------------
        # Save model locally
        # --------------------------------------------------

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = MODEL_DIR / "valve_classifier.joblib"

        joblib.dump(
            model,
            model_path,
        )

        print(f"\nModel saved to: {model_path}")

        # --------------------------------------------------
        # Log model to MLflow
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        print("Model logged to MLflow.")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="pickle",
        )

        print("Dataset logged to MLflow.")


if __name__ == "__main__":
    main()
