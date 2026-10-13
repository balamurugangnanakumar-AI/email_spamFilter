# Lifecycle stage 4 — Data Preparation (full dataset)
import pathlib as pl
import pandas as pd
from src.data.preprocess import clean_batch

PROJECT_ROOT = pl.Path(__file__).resolve().parents[2]
 
def process_dataset():
 
    print("Script Started")
 
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "spamhamdata.csv",
        sep="\t",
        header=None,
        names=["label", "email_text"]
    )
 
    print("Dataset Loaded")
 
    # Lowercasing + regex are fast as vectorized pandas ops,
    # so we do them here and leave only tokenizing/lemmatizing to spaCy.
 
    texts = df["email_text"].fillna("").astype(str).str.lower()
    texts = texts.str.replace(r"http\S+", "", regex=True)
    texts = texts.str.replace(r"[^a-zA-Z ]", "", regex=True)
 
    df["clean_email"] = clean_batch(texts)
 
    print("Cleaning Completed")
 
    df.to_csv(
        PROJECT_ROOT / "data" / "processed" / "cleaned_emails.csv",
        index=False
    )
 
    print("File Saved")
