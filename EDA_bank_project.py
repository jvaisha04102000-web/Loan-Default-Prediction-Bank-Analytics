import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("bank_loan_dataset.csv")
print(df.head())
print(df.info())
print(df.shape)

print(df.isnull().sum())
print(df.describe())

sns.countplot(x="Loan_Default", data=df)
plt.title("Loan Default Distribution")
plt.show()

sns.histplot(df["Age"], bins=20)
plt.title("Customer Age Distribution")
plt.show()

sns.boxplot(x="Employment_Type", y="Loan_Amount", data=df)
plt.title("Loan Amount by Employment Type")
plt.show()
