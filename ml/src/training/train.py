from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("ml/data/processed/hydraulic_features.csv")

TARGET = "valve_condition"
LABELS = [73, 80, 90, 100]

MIN_F1 = 0.90

REGISTERED_MODEL_NAME = "hydraulic-valve-classifier"
MODEL_ALIAS = "champion"


def main():

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

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

    # ---------------------------------------------------------
    # Train / test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ---------------------------------------------------------
    # MLflow experiment
    # ---------------------------------------------------------

    mlflow.set_experiment("hydraulic-valve-condition")

    experiments = [
        {
            "name": "rf_100",
            "n_estimators": 100,
            "max_depth": None,
        },
        {
            "name": "rf_300",
            "n_estimators": 300,
            "max_depth": None,
        },
        {
            "name": "rf_500",
            "n_estimators": 500,
            "max_depth": None,
        },
        {
            "name": "rf_depth_10",
            "n_estimators": 300,
            "max_depth": 10,
        },
        {
            "name": "rf_depth_20",
            "n_estimators": 300,
            "max_depth": 20,
        },
    ]

    # ---------------------------------------------------------
    # Best model tracking
    # ---------------------------------------------------------

    best_model = None
    best_score = -1
    best_config = None
    best_run_id = None
    best_model_id = None

    # ---------------------------------------------------------
    # Train experiments
    # ---------------------------------------------------------

    for config in experiments:

        print("\n" + "=" * 50)
        print(f"Training: {config['name']}")
        print("=" * 50)

        with mlflow.start_run(run_name=config["name"]):

            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

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

            accuracy = accuracy_score(
                y_test,
                y_pred,
            )

            # -------------------------------------------------
            # Log parameters
            # -------------------------------------------------

            mlflow.log_param(
                "model",
                "RandomForest",
            )

            mlflow.log_param(
                "n_estimators",
                config["n_estimators"],
            )

            mlflow.log_param(
                "max_depth",
                config["max_depth"],
            )

            mlflow.log_param(
                "class_weight",
                "balanced",
            )

            mlflow.log_param(
                "random_state",
                42,
            )

            mlflow.log_param(
                "num_features",
                X.shape[1],
            )

            mlflow.log_param(
                "num_samples",
                X.shape[0],
            )

            # -------------------------------------------------
            # Log metrics
            # -------------------------------------------------

            mlflow.log_metric(
                "weighted_f1",
                weighted_f1,
            )

            mlflow.log_metric(
                "macro_f1",
                macro_f1,
            )

            mlflow.log_metric(
                "accuracy",
                accuracy,
            )

            print(f"Weighted F1: {weighted_f1:.4f}")
            print(f"Macro F1:    {macro_f1:.4f}")
            print(f"Accuracy:    {accuracy:.4f}")

            # -------------------------------------------------
            # Log model
            # -------------------------------------------------

            logged_model = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="pickle",
            )

            # -------------------------------------------------
            # Track best model
            # -------------------------------------------------

            if weighted_f1 > best_score:

                best_score = weighted_f1
                best_model = model
                best_config = config.copy()

                best_run_id = (
                    mlflow.active_run().info.run_id
                )

                best_model_id = logged_model.model_id

    # ---------------------------------------------------------
    # Best model summary
    # ---------------------------------------------------------

    print("\n" + "=" * 50)
    print("BEST MODEL")
    print("=" * 50)

    print(
        f"Configuration: {best_config['name']}"
    )

    print(
        f"Trees:         {best_config['n_estimators']}"
    )

    print(
        f"Max depth:     {best_config['max_depth']}"
    )

    print(
        f"Weighted F1:   {best_score:.4f}"
    )

    print(
        f"Run ID:        {best_run_id}"
    )

    print(
        f"Logged Model:  {best_model_id}"
    )

    # ---------------------------------------------------------
    # Quality gate
    # ---------------------------------------------------------

    if best_score < MIN_F1:

        raise RuntimeError(
            f"Model failed quality gate. "
            f"Weighted F1={best_score:.4f}, "
            f"minimum required={MIN_F1:.4f}"
        )

    print("\nModel passed quality gate.")

    # ---------------------------------------------------------
    # Register model
    # ---------------------------------------------------------

    print("\nRegistering best model...")

    client = MlflowClient()

    try:

        client.create_registered_model(
            name=REGISTERED_MODEL_NAME
        )

        print(
            f"Created registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

    except Exception:

        print(
            f"Registered model already exists: "
            f"{REGISTERED_MODEL_NAME}"
        )

    # ---------------------------------------------------------
    # Create model version from logged model
    # ---------------------------------------------------------

    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=f"models:/{best_model_id}",
        run_id=best_run_id,
    )

    print(
        f"Registered model version: "
        f"{model_version.version}"
    )

    # ---------------------------------------------------------
    # Promote model to champion
    # ---------------------------------------------------------

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=model_version.version,
    )

    print(
        f"Model version {model_version.version} "
        f"is now the '{MODEL_ALIAS}' model."
    )

    # ---------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------

    y_pred = best_model.predict(X_test)

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

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)

    print(
        f"Model:       {REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version:     {model_version.version}"
    )

    print(
        f"Alias:       @{MODEL_ALIAS}"
    )

    print(
        f"Weighted F1: {best_score:.4f}"
    )

    print(
        f"Run ID:      {best_run_id}"
    )

    print(
        f"Model ID:    {best_model_id}"
    )


if __name__ == "__main__":
    main()