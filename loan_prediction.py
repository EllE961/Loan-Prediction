import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
for col in cat_cols:
    train_df[col].fillna("Unknown", inplace=True)
    test_df[col].fillna("Unknown", inplace=True)
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History", "ApplicantIncome", "CoapplicantIncome"]
for col in num_cols:
    imputer = SimpleImputer(strategy="median")
    train_df[col] = imputer.fit_transform(train_df[[col]])
    test_df[col] = imputer.transform(test_df[[col]])

train_df.dropna(subset=["Loan_Status"], inplace=True)
train_df["Loan_Status"] = train_df["Loan_Status"].map({"N": 0, "Y": 1})

train_df["Total_Income"] = train_df["ApplicantIncome"] + train_df["CoapplicantIncome"]
train_df["Income_to_Loan"] = train_df["Total_Income"] / (train_df["LoanAmount"] + 1)

X = train_df.drop(["Loan_ID", "Loan_Status"], axis=1, errors="ignore")
y = train_df["Loan_Status"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10]
}
model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Best Params:", grid.best_params_)
print("Accuracy:", acc)
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=["No", "Yes"]))

cm = confusion_matrix(y_val, y_pred)

sns.set_style("whitegrid")
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

loan_counts = train_df["Loan_Status"].value_counts()
sns.barplot(x=loan_counts.index.map({0: "No", 1: "Yes"}), y=loan_counts.values)
plt.title("Loan_Status Distribution")
plt.ylabel("Count")
plt.show()

sns.histplot(train_df["LoanAmount"], kde=True, bins=30)
plt.title("Loan Amount Distribution")
plt.xlabel("LoanAmount")
plt.show()

sns.scatterplot(data=train_df, x="ApplicantIncome", y="CoapplicantIncome", hue="Loan_Status")
plt.title("Income Scatter by Loan_Status")
plt.show()

sns.boxplot(data=train_df, x="Loan_Status", y="Income_to_Loan")
plt.title("Income to Loan Ratio vs Loan_Status")
plt.show()
