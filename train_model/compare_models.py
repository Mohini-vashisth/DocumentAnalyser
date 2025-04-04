import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os
import sys

def load_data(path):
    df = pd.read_csv(path)

    # Fix label name mismatch
    if "label" in df.columns and "risk" not in df.columns:
        df = df.rename(columns={"label": "risk"})

    df = df.drop_duplicates(subset="clause")
    print(f"\n📊 Evaluating on: {os.path.basename(path)}")
    print("🧪 Class Distribution:\n", df["risk"].value_counts())
    return df["clause"].tolist(), df["risk"].tolist()

def load_model(model_folder):
    encoder = SentenceTransformer(f"{model_folder}/sentence_encoder")
    clf = joblib.load(f"{model_folder}/clause_classifier.pkl")
    return encoder, clf

def evaluate_model(model_name, model_folder, X_texts, y_true):
    encoder, clf = load_model(model_folder)
    X_vecs = encoder.encode(X_texts)
    y_pred = clf.predict(X_vecs)

    print(f"\n📌 Evaluation for model: {model_name}")
    print(classification_report(y_true, y_pred, digits=3))

if __name__ == "__main__":
    # Evaluation dataset
    X, y = load_data("data/legal_clauses_labeled.csv")  # or ledgar_clauses.csv or combined

    # Evaluate CUAD model
    evaluate_model("CUAD", "saved_model", X, y)

    # Evaluate LEDGAR model
    evaluate_model("LEDGAR", "saved_model_ledgar_model", X, y)