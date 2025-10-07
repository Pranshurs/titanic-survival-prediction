import pandas as pd
import joblib
import yaml
from titanic_survival.src.features.preprocess import preprocess


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_submission():
    print("🚀 Starting prediction script...")
    config = load_config()

    # Load model
    model_path = config["model"]["output_path"]
    print(f"📦 Loading model from {model_path}")
    model = joblib.load(model_path)

    # Load test data
    test_path = config["data"]["test_path"]
    print(f"📄 Reading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    # Keep PassengerId for submission
    passenger_ids = test_df["PassengerId"]

    # Preprocess same as training
    X_test = preprocess(test_df)

    # Predict
    preds = model.predict(X_test)

    # Build submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds
    })

    output_path = "titanic_survival/models/submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"✅ Submission file saved at {output_path}")


if __name__ == "__main__":
    generate_submission()