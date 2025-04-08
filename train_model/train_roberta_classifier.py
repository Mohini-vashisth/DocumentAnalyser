import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data():
    df = pd.read_csv(os.path.join("data", "legal_clauses_labeled.csv")).rename(columns={"label": "risk"})
    df.drop_duplicates(subset=["clause"], inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("\n📊 Label Distribution:")
    print(df["risk"].value_counts())
    return df["clause"].tolist(), df["risk"].tolist()

def encode_clauses(model, tokenizer, clauses, batch_size=16):
    """
    Encodes clauses in batches to avoid OOM errors.
    """
    all_embeddings = []
    model.eval()

    for i in range(0, len(clauses), batch_size):
        batch = clauses[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings).numpy()

def train_model(X_texts, y_labels):
    print("\n🧠 Loading RoBERTa...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")

    print("🔍 Encoding clauses...")
    X_vectors = encode_clauses(model, tokenizer, X_texts)

    print("📂 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_vectors, y_labels, test_size=0.2, random_state=42)

    print("🤖 Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("\n✅ Model trained.")
    y_pred = clf.predict(X_test)
    print("\n📊 Evaluation Report:")
    print(classification_report(y_test, y_pred, digits=3))

    return model, tokenizer, clf

def save_model(model, tokenizer, classifier):
    os.makedirs("saved_model_roberta", exist_ok=True)
    joblib.dump(classifier, os.path.join("saved_model_roberta", "clause_classifier.pkl"))
    model.save_pretrained("saved_model_roberta")
    tokenizer.save_pretrained("saved_model_roberta")
    print("\n✅ Models saved to 'saved_model_roberta/'")

if __name__ == "__main__":
    X_texts, y_labels = load_data()
    encoder, tokenizer, classifier = train_model(X_texts, y_labels)
    save_model(encoder, tokenizer, classifier)