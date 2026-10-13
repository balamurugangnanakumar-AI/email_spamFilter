#load data from the raw CSV file
# Lifecycle stage 3 — Data Understanding
 
import pandas as pd
def load_data(path="pythonFiles\\AgenticAI\\email_spamFilter\\data\\raw\\spamhamdata.csv"):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"]
    )


