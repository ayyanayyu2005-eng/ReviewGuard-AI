
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Read dataset

reviews = []

DATA_DIR = Path("data/extracted")

for file in DATA_DIR.rglob("*.txt"):

    path = str(file).lower()

    if "deceptive_from_mturk" in path:
        label = "Fake"

    elif "truthful_from_web" in path or "truthful_from_tripadvisor" in path:
        label = "Genuine"

    else:
        continue

    text = file.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if text:
        reviews.append({
            "review": text,
            "label": label
        })

# Step 2: Create DataFrame

df = pd.DataFrame(reviews)
df.to_csv("reviews_dataset.csv", index=False)

print("Total Reviews:", len(df))
print("\nDataset loaded successfully!")

# Step 3: Separate input and output

X = df["review"]
y = df["label"]

# Step 4: Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Reviews:", len(X_train))
print("Testing Reviews:", len(X_test))

# Step 5: Create ML model

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Step 6: Train the model

print("\nTraining AI Model...")

model.fit(X_train, y_train)

print("Model training completed!")

# Step 7: Test the model

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Step 8: Save trained model

with open("fake_review_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
