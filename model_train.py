import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Loading the dataset
df = pd.read_csv("heart_failure_clinical_records_dataset (1).csv")

# Prepareing data
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model for prediction
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

#Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
