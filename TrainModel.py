import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load your extracted coordinate database
print("Loading asl_data.csv...")
df = pd.read_csv("asl_data.csv")

# Separate the 42 hand coordinates (X) from the letter labels (y)
X = df.drop("label", axis=1)
y = df["label"]

# 2. Split data: 80% for studying (training), 20% for a final exam (testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

# 3. Create and train the Random Forest classifier
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Test the model on the 20% exam data it has never seen before
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 5. Save the trained model to a file so slt.py can use it
with open("model.p", "wb") as f:
    pickle.dump({"model": model}, f)

print("Success! Trained model saved as model.p")