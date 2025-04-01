import os
import re
import joblib
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import fitz  # PyMuPDF
from groq import Groq

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        print(f"💥 Error reading PDF file: {e}")
        return ""

def split_into_clauses(text):
    """
    Splits extracted text into clauses based on basic rules.
    """
    clauses = re.split(r'\n\s*\d+\.|\n\n|\.\s', text)
    return [cl.strip() for cl in clauses if len(cl.strip()) > 20]

def load_models(model_folder="saved_model"):
    """
    Loads the SentenceTransformer model and the Logistic Regression classifier.
    """
    try:
        encoder = SentenceTransformer(os.path.join(model_folder, "sentence_encoder"))
        clf = joblib.load(os.path.join(model_folder, "clause_classifier.pkl"))
        return encoder, clf
    except Exception as e:
        print(f"💥 Error loading models from '{model_folder}': {e}")
        return None, None

def explain_clause(clause):
    """
    Uses the Groq API to generate a plain English explanation for a risky clause.
    """
    prompt = f"""
    You are a legal assistant helping non-lawyers understand legal contracts.

    Explain the following clause in simple, non-legal English. If it might be risky or one-sided, say why:

    Clause:
    \"\"\"
    {clause}
    \"\"\"
    """

    try:
        response = groq_client.chat.completions.create(
            model="mistral-saba-24b",
            messages=[
                {"role": "system", "content": "You simplify and explain legal clauses clearly for non-lawyers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"💥 Error: {str(e)}"

def analyze_document(file_path, model_folder="saved_model"):
    """
    Analyzes a legal document, detects risky clauses, and provides explanations.
    """
    if file_path.endswith(".pdf"):
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(file_path)
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"💥 Error reading file '{file_path}': {e}")
            return

    clauses = split_into_clauses(text)
    encoder, clf = load_models(model_folder)

    if encoder is None or clf is None:
        print("❌ Failed to load models. Please check your model folder path.")
        return

    print(f"\nFound {len(clauses)} clauses. Analyzing...\n")
    clause_vectors = encoder.encode(clauses)
    predictions = clf.predict(clause_vectors)

    for clause, label in zip(clauses, predictions):
        status = "RISKY" if label == "risky" else "Safe"
        explanation = explain_clause(clause) if label == "risky" else ""
        print(f"\n--- {status} ---\n{clause}")
        if explanation:
            print(f"Explanation: {explanation}")

if __name__ == "__main__":
    # Use relative path for the sample file
    test_file = os.path.join(os.getcwd(), "Sample.pdf")
    if not os.path.exists(test_file):
        print(f"❌ File '{test_file}' not found. Add a contract text file first.")
    else:
        analyze_document(test_file, model_folder="saved_model")