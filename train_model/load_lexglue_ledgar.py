from datasets import load_dataset
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_ledgar(save_path="data/ledgar_clauses.csv", num_samples=5000):
    """
    Loads LEDGAR dataset from HuggingFace's lex_glue repository,
    processes it into labeled clauses, and saves as a CSV file.

    Args:
        save_path (str): Path to save the processed CSV file.
        num_samples (int): Number of samples to select from the training set.
    """
    try:
        dataset = load_dataset("lex_glue", "ledgar")
    except Exception as e:
        print(f"❌ Error loading LEDGAR dataset: {e}")
        return

    # Select a subset of the dataset for faster processing
    train_data = dataset["train"].select(range(num_samples))  

    # Get label ID → name mapping
    label_id2name = train_data.features["label"].int2str

    # Extract clause text and string labels
    clauses = train_data["text"]
    labels = [label_id2name(label_id) for label_id in train_data["label"]]

    # Create DataFrame
    df = pd.DataFrame({"clause": clauses, "label": labels})

    print("\n✅ All available clause types in LEDGAR:")
    print(df["label"].value_counts())

    # Define risky labels (Make this configurable)
    risky_labels = [
        "Limitation of Liability",
        "Termination",
        "Confidentiality",
        "Indemnification",
        "Force Majeure",
        "Dispute Resolution",
        "Exclusivity",
        "Non-Compete",
        "Severability",
        "Arbitration"
    ]

    # Apply labeling
    df["risk"] = df["label"].apply(lambda l: "risky" if l.strip() in risky_labels else "safe")

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the labeled dataset
    df[["clause", "risk"]].to_csv(save_path, index=False)
    print(f"✅ Data saved successfully to {save_path}")

if __name__ == "__main__":
    load_ledgar()