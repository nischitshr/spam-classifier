import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report



possible_files = ["SMSSpamCollection", "SMSSpamCollection.txt", "spam.csv"]
df = None

for fname in possible_files:
    try:
        df = pd.read_csv(
            fname,
            sep="\t",              # tab-separated
            header=None,           # no header row in file
            names=["label", "message"],
            encoding="latin-1"     # avoids weird encoding errors
        )
        print(f"✅ Loaded: {fname}")
        break
    except FileNotFoundError:
        continue

if df is None:
    raise SystemExit("❌ File not found. Make sure SMSSpamCollection is in this folder.")

print("Shape:", df.shape)
print(df.head())
print("\nClass counts:\n", df["label"].value_counts())

# ---------- 2. SPLIT ----------
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ---------- 3. VECTORIZE ----------
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# ---------- 4. TRAIN ----------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ---------- 5. EVALUATE ----------
y_pred = model.predict(X_test_vec)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nReport:")
print(classification_report(y_test, y_pred))

# ---------- 6. PREDICT NEW MESSAGES ----------
print("\n--- Try your own messages ---")
tests = [
    "Congratulations! You won a free trip to Paris, claim now",
    "Hey, are we meeting for coffee today?",
    "URGENT: your bank account needs verification, click here",
    "I'll call you back in 10 minutes"
]
for msg in tests:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    print(f"[{pred.upper():4}] ({prob*100:.1f}%)  {msg}")
    