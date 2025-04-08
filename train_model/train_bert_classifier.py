import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import os


def load_data():
    """
    Loads the CUAD and LEDGAR datasets, combines them, and returns clauses and labels.
    
    Returns:
        list: Clauses (text).
        list: Corresponding labels ('risky' or 'safe').
    """
    # Load CUAD dataset (labeled earlier) from data folder
    df_cuad = pd.read_csv(os.path.join("data", "legal_clauses_labeled.csv")).rename(columns={"label": "risk"})

    # Drop duplicate clauses
    df_cuad.drop_duplicates(subset=["clause"], inplace=True)

    # Shuffle the combined dataset for training robustness
    df_cuad = df_cuad.sample(frac=1, random_state=42).reset_index(drop=True)

    # Display the label distribution
    print("\n📊 Label Distribution:")
    print(df_cuad["risk"].value_counts())

    # Return clauses and labels as lists
    return df_cuad["clause"].tolist(), df_cuad["risk"].tolist()
    
def train_model(X_texts, y_labels):
    """
    Trains a Logistic Regression classifier using embeddings from SentenceTransformer.
    
    Args:
        X_texts (list): Clauses as text.
        y_labels (list): Corresponding labels ('risky' or 'safe').
    
    Returns:
        model: Trained SentenceTransformer model.
        clf: Trained Logistic Regression classifier.
    """
    print("\n Loading SentenceTransformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(" Encoding clauses into embeddings...")
    X_vectors = model.encode(X_texts)

    print(" Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_vectors, y_labels, test_size=0.2, random_state=42)

    print(" Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("\n Model trained successfully.")
    y_pred = clf.predict(X_test)
    print("\n Evaluation Report:")
    print(classification_report(y_test, y_pred, digits=3))

    return model, clf

def save_model(embedding_model, classifier_model):
    """
    Saves the trained embedding model and classifier to the saved_model folder.
    
    Args:
        embedding_model: Trained SentenceTransformer model.
        classifier_model: Trained Logistic Regression classifier.
    """
    # Ensure the directory exists
    os.makedirs("saved_model", exist_ok=True)
    
    # Save the encoder model
    embedding_model.save(os.path.join("saved_model", "sentence_encoder"))

    # Save the classifier model
    joblib.dump(classifier_model, os.path.join("saved_model", "clause_classifier.pkl"))
    print("\n Models saved successfully to 'saved_model/' directory.")

if __name__ == "__main__":
    X_texts, y_labels = load_data()
    encoder, classifier = train_model(X_texts, y_labels)
    save_model(encoder, classifier)