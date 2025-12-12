import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score


# Load dataset
df = pd.read_csv("train.csv")

# Select useful columns
cols = [
    "Type", "Breed1", "Age", "Color1", "Color2", "Color3",
    "Gender", "Health", "PhotoAmt", "State", "AdoptionSpeed"
]

df = df[cols].dropna()


# Age Histogram
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=40, color="steelblue")
plt.title("Distribution of Pet Age")
plt.xlabel("Age (months)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("age_histogram.png")
plt.close()


# State counts
state_counts = df["State"].value_counts()

plt.figure(figsize=(8,5))
state_counts.head(15).plot(kind="bar", color="coral")
plt.title("Pet Listings by State")
plt.xlabel("State Code")
plt.ylabel("Number of Pets")
plt.tight_layout()
plt.savefig("state_pet_counts.png")
plt.close()


# Prepare Features for Modeling
X = df[["Type", "Breed1", "Age", "Color1", "Color2", "Color3",
        "Gender", "Health", "PhotoAmt"]]

y = df["AdoptionSpeed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Logistic Regression Model
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))


# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# Confusion Matrix
plt.figure(figsize=(6,5))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")
plt.close()


# Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=feature_names, color="teal")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.close()

print("\nAll plots have been saved:")
print("- age_histogram.png")
print("- state_pet_counts.png")
print("- confusion_matrix_rf.png")
print("- feature_importance_rf.png")
print("Your final project code ran successfully!")
