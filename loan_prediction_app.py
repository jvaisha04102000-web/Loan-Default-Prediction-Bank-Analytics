import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("bank_loan_dataset.csv")

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

X = df.drop(["Customer_ID","Loan_Default"], axis=1)
y = df["Loan_Default"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.title("Bank Loan Default Prediction")

age = st.number_input("Customer Age")
income = st.number_input("Monthly Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")

employment = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
previous_loan =st.selectbox("Previous_Loan", ["No","Yes"])

employment_map = {"Salaried":0, "Self-Employed":1, "Business":2}
previous_map = {"No":0, "Yes":1}

employment = employment_map[employment]
previous_loan = previous_map[previous_loan]

if st.button("Predict Loan Default"):

    input_data = [[age, income, loan_amount, credit_score, employment, previous_loan]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer likely to DEFAULT loan")
    else:
        st.success("Customer likely to REPAY loan")
