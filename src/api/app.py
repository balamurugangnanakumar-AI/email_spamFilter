# Lifecycle stage 10 — Model Deployment (hand-off)
 
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import importlib.util
from pathlib import Path

from src.data.preprocess import clean_text

app = FastAPI()

module_path = Path(__file__).resolve().parents[1] / "quarantine" / "store.py"
if module_path.exists():
    spec = importlib.util.spec_from_file_location("quarantine_store", module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        quarantine_email = module.quarantine_email
    else:
        quarantine_email = None
else:
    quarantine_email = None

if quarantine_email is None:
    def quarantine_email(*args, **kwargs):
        return None
model = joblib.load("models/spam_model.pkl")
 
vectorizer = joblib.load("models/count_vectorizer.pkl")
 
 
class Email(BaseModel):
    text: str
 
 
@app.post("/predict")
def predict(email: Email):
 
    X = vectorizer.transform([clean_text(email.text)])
 
    label = model.predict(X)[0]
 
    if label == "spam":
        quarantine_email(email.text, label)
 
    return {"label": label}
