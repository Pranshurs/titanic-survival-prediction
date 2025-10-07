import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    df = pd.read_csv(config["data"]["train_path"])
    return df

def split_data(df, config):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return train_test_split(X, y, test_size=config["training"]["test_size"], random_state=config["training"]["random_state"])