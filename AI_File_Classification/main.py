
# AI-Based File Classification System
# Author: Zeel Sheta

import os
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- TRAINING DATA ----------------
texts = [
    "salary slip employee company",
    "invoice bill payment gst",
    "python programming loops functions",
    "machine learning ai neural network",
    "song lyrics music singer",
    "resume cv skills education",
    "bank statement transaction debit credit"
]

labels = [
    "Office",
    "Bills",
    "Study",
    "Study",
    "Music_Text",
    "Resumes",
    "Finance"
]

# ---------------- MODEL TRAINING ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# ---------------- PREDICTION FUNCTION ----------------
def predict_category(text):
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]

# ---------------- FILE ORGANIZATION ----------------
path = "YOUR_FOLDER_PATH_HERE"   # Change this

for file in os.listdir(path):
    file_path = os.path.join(path, file)

    if file.endswith(".txt"):
        with open(file_path, "r", errors="ignore") as f:
            content = f.read()

        category = predict_category(content)
        category_path = os.path.join(path, category)
        os.makedirs(category_path, exist_ok=True)

        shutil.move(file_path, os.path.join(category_path, file))

print("AI-based file organization completed successfully.")
