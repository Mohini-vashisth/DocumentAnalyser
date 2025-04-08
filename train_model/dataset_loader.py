import json
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_cuad_dataset(json_path):
    """
    Loads the CUAD dataset from a given JSON file and processes it into a DataFrame.
    Args:
        json_path (str): Path to the CUAD JSON file.
    Returns:
        pd.DataFrame: DataFrame containing the extracted clauses and questions.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f" File not found: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f" Invalid JSON format in file: {json_path}")

    rows = []
    for item in data["data"]:
        context = item["paragraphs"][0]["context"]
        for qa in item["paragraphs"][0]["qas"]:
            question = qa["question"]
            for answer in qa["answers"]:
                answer_text = answer["text"]
                rows.append({
                    "context": context,
                    "question": question,
                    "clause": answer_text
                })

    df = pd.DataFrame(rows)
    return df

def clean_and_label(df, risk_keywords=None):
    """
    Cleans the clause text and labels them as risky or safe based on keywords.
    Args:
        df (pd.DataFrame): DataFrame containing clauses to be labeled.
        risk_keywords (list): List of keywords considered risky.
    Returns:
        pd.DataFrame: DataFrame with labeled clauses.
    """
    df["clause"] = df["clause"].str.strip().str.replace(r"\s+", " ", regex=True)
    
    # Use default keywords if none provided
    if risk_keywords is None:
        risk_keywords = ['liability', 'indemnify', 'termination', 'penalty', 'breach', 'damages', 'loss']

    def label_risk(clause):
        for word in risk_keywords:
            if word.lower() in clause.lower():
                return "risky"
        return "safe"

    df["label"] = df["clause"].apply(label_risk)
    return df

if __name__ == "__main__":
    # Use relative path for the CUAD JSON file
    json_path = os.path.join("cuad", "data", "CUADv1.json")

    try:
        # Load and clean
        df = load_cuad_dataset(json_path)
        df = clean_and_label(df)
        
        # Ensure the data directory exists
        os.makedirs("data", exist_ok=True)

        # Save the labeled data to CSV
        output_path = os.path.join("data", "legal_clauses_labeled.csv")
        df[["clause", "label"]].to_csv(output_path, index=False)
        print(f" File saved successfully at {output_path}")
        
    except Exception as e:
        print(f" Error: {e}")