from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import yaml

from titanic_survival.src.data.load_data import load_data, split_data
from titanic_survival.src.features.preprocess import preprocess

print("🔍 Debug: train.py script started running")

def train_model():
    print("🔍 Debug: train.py script started running")

    import os
    print("📂 Current working dir:", os.getcwd())
    print("📁 Files in this dir:", os.listdir())

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("✅ Config loaded successfully")

    print("📄 Looking for:", config["data"]["train_path"])

    df = load_data(config)
    print("✅ Data loaded, shape:", df.shape)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    df = load_data(config)
    print("✅ Data loaded, shape:", df.shape)
    X_train, X_test, y_train, y_test = split_data(df, config)
    X_train, X_test = preprocess(X_train), preprocess(X_test)

    if config["model"]["type"] == "random_forest":
        model = RandomForestClassifier(random_state=config["training"]["random_state"])
    else:
        model = LogisticRegression(max_iter=500)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"✅ Model trained with accuracy: {acc:.4f}")

    joblib.dump(model, config["model"]["output_path"])
    print(f"💾 Model saved at {config['model']['output_path']}")
    
if __name__ == "__main__":
    train_model()