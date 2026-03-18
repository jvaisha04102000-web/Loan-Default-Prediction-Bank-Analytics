import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("bank_loan_dataset.csv")
print(df.head())

df["Previous_Loan"] = df["Previous_Loan"].str.strip().str.capitalize()
df["Loan_Default"] = df["Loan_Default"].str.strip().str.capitalize()
df["Employment_Type"] = df["Employment_Type"].str.strip()

df["Employment_Type"] = df["Employment_Type"].map({
    "Salaried":0,
    "Self-Employed":1,
    "Business":2
})

df["Previous_Loan"] = df["Previous_Loan"].map({
    "No":0,
    "Yes":1
})

df["Loan_Default"] = df["Loan_Default"].map({
    "No":0,
    "Yes":1
})

# Define Features and Target
X = df.drop("Loan_Default", axis=1)
y = df["Loan_Default"]

# Split Traning and Testing Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)