import pandas as pd
import numpy as np
#number of records
num_rows = 1000

np.random.seed(42)

data = {
    "Customer_ID": range(1001, 1001 + num_rows),
    "Age" : np.random.randint(21, 60, num_rows),
    "Monthly_Income": np.random.randint(20000, 100000, num_rows),
    "Loan_Amount": np.random.randint(50000, 500000, num_rows),
    "Credit_Score": np.random.randint(300, 850, num_rows),
    "Employment_Type": np.random.choice(["Salaried", "Self-Employed", "Business"], num_rows),
    "Previous_Loan" : np.random.choice(["Yes", "No"], num_rows),
    "Loan_Default" : np.random.choice(["Yes", "NO"], num_rows)
}

df = pd.DataFrame(data)

df.to_csv("bank_loan_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print(df.shape)