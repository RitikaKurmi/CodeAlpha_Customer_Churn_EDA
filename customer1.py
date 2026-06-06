
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


df =  pd.read_csv('Customer Churn.csv')
print(df)
print(df.info())

df["TotalCharges"] = df["TotalCharges"].replace(" ","0")
df["TotalCharges"] = df["TotalCharges"].astype("float")
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.duplicated().sum())
print(df["customerID"].duplicated().sum())


def conv(value):
    if value == 1:
        return "yes"
    else :
      return "no"


df["SeniorCitizen"] = df["SeniorCitizen"].apply(conv)

# graph1
churn = df["Churn"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(churn.index, churn.values)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.show()

# graph2
gender = df["gender"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(gender.index, gender.values)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Customers")

plt.show()

# graph3
pd.crosstab(df["gender"], df["Churn"]).plot(kind="bar")

plt.title("Gender vs Churn")
plt.xlabel("Gender")
plt.ylabel("Customers")

plt.show()

#graph4
pd.crosstab(df["SeniorCitizen"], df["Churn"]).plot(kind="bar")

plt.title("Senior Citizen vs Churn")
plt.xlabel("Senior Citizen")
plt.ylabel("Customers")

plt.show()

#graph5
pd.crosstab(df["Contract"], df["Churn"]).plot(kind="bar")

plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Customers")

plt.xticks(rotation=20)

plt.show()

#graph6
internet = df["InternetService"].value_counts()

plt.figure(figsize=(6,4))

plt.bar(internet.index, internet.values)

plt.title("Internet Service Distribution")
plt.xlabel("Internet Service")
plt.ylabel("Customers")

plt.show()

#graph7
pd.crosstab(df["InternetService"], df["Churn"]).plot(kind="bar")

plt.title("Internet Service vs Churn")
plt.xlabel("Internet Service")
plt.ylabel("Customers")

plt.show()


#graph8
pd.crosstab(df["PaymentMethod"], df["Churn"]).plot(kind="bar")

plt.title("Payment Method vs Churn")
plt.xlabel("Payment Method")
plt.ylabel("Customers")

plt.xticks(rotation=45)

plt.show()


#graph9
plt.figure(figsize=(8,5))

plt.hist(df["MonthlyCharges"], bins=20)

plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Frequency")

plt.show()

#graph10
yes = df[df["Churn"]=="Yes"]["MonthlyCharges"]
no = df[df["Churn"]=="No"]["MonthlyCharges"]

plt.figure(figsize=(7,5))

plt.boxplot([yes, no])

plt.xticks([1,2], ["Yes","No"])

plt.title("Monthly Charges vs Churn")
plt.ylabel("Monthly Charges")

plt.show()