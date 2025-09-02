import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
#1.1

df = pd.read_csv("retail_sales_dataset_custom.csv")
print("First 10 rows:")
print(df.head(10))

print("\nLast 10 rows:")
print(df.tail(10))

print( df.shape)
print("\nColumn names:")
print(df.columns)

print("\nRandom Sample:")
print(df.sample(5))

#1.2

print(df.info())
print(df.describe())
print(df.nunique())
print(df["Category"].mode()[0])
print("Earliest date:", df["Date"].min())
print("Latest date:", df["Date"].max())
print("Average quantity per transaction:", df["Quantity"].mean())
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
print(df["Date"].dt.year.value_counts())

#1.3

print(df.isnull().sum())
df_original = df.copy()
df = df.fillna(df.mean(numeric_only=True))
df = df.fillna(df.mode().iloc[0])
changed_rows = ((df_original.isnull().sum(axis=1) > 0) & (df.isnull().sum(axis=1) == 0)).sum()
print("Number of rows changed:", changed_rows)
df = df.drop_duplicates()
df = df.drop(columns=['Holiday_Flag'])

#1.4

df["Category"] = df["Category"].astype("category")
df["Payment_Method"] = df["Payment_Method"].astype("category")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
print(df.info())

#1.5

df["Total_Sales"] = df["Price"] * df["Quantity"]
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["Day_of_Week"] = df["Date"].dt.day_name()
df["Is_Weekend"] = df["Day_of_Week"].isin(["Saturday", "Sunday"])
df["Price_per_Quantity"] = df["Total_Sales"] / df["Quantity"]
df["Sale_Level"] = np.where(df["Total_Sales"] > 500, "High Sale", "Low Sale")
print(df.head())

#1.6

total_revenue = df["Total_Sales"].sum()
print("Total Revenue:", total_revenue)
average_order_value = df["Total_Sales"].mean()
print("Average Order Value:", average_order_value)
Max_transaction = df["Total_Sales"].max()
Low_transaction = df["Total_Sales"].min()
print("Highest Transaction:", Max_transaction)
print("Lowest Transaction:", Low_transaction)
category_sales = df.groupby("Category")["Total_Sales"].sum()
most_sales_category = category_sales.idxmax()
most_sales_value = category_sales.max()
print("Category with most sales:", most_sales_category)
print("Total sales:", most_sales_value)
revenue_per_store = df.groupby("Store")["Total_Sales"].sum()
highest_store = revenue_per_store.idxmax()
lowest_store = revenue_per_store.idxmin()
print("Highest Revenue Store:", highest_store)
print("Lowest Revenue Store:", lowest_store)
most_common_payment = df["Payment_Method"].mode()[0]
print("Most Common Payment Method:", most_common_payment)
avg_weekend_revenue = df[df["Is_Weekend"] == True]["Total_Sales"].mean()
print("Average Weekend Revenue:", avg_weekend_revenue)

#1.7
pd.set_option("display.float_format", "{:,.0f}".format)
sales_per_month = df.groupby(["Month","Year"])["Total_Sales"].sum()
print("sales_per_month are :",sales_per_month)
sales_per_category = df.groupby("Category")["Total_Sales"].sum()
print("most_sales_per_category is:",sales_per_category)
sales_per_payment = df.groupby("Payment_Method")["Total_Sales"].sum()
print("sales_per_payment are :",sales_per_payment)
avg_price_per_category = df.groupby("Category")["Price"].mean()
print("avg_price_per_category is :",avg_price_per_category)
top_3_products = df.groupby("Product")["Total_Sales"].sum().sort_values(ascending=False).head(3)
print("top_3_products are:",top_3_products)
bottom_3_products = df.groupby("Product")["Total_Sales"].sum().sort_values().head(3)
print("bottom_3_products are :",bottom_3_products)
transactions_per_month = df.groupby("Month")["Total_Sales"].count()
print("transactions_per_month are :",transactions_per_month)
revenue_per_store_per_year = df.groupby(["Store", "Year"])["Total_Sales"].sum()
print("revenue_per_store_per_year is :",revenue_per_store_per_year)
avg_quantity_payment = df.groupby("Payment_Method")["Quantity"].mean()
print("avg_quantity_payment is :",avg_quantity_payment)

#1.8

df["Sales"] = df["Price"] * df["Quantity"]
print(df.groupby("Product")["Sales"].sum().sort_values(ascending=False))
print(df[df["Sales"] > 500])
print(df[df["Payment_Method"] == "Credit Card"])
print(df[df["Category"] == "Electronics"])

#1.9

mean_sales = df["Total_Sales"].mean()
median_sales = df["Total_Sales"].median()
std_sales = df["Total_Sales"].std()
max_sales = df["Total_Sales"].max()
min_sales = df["Total_Sales"].min()
print("Mean Total Sales:", mean_sales)
print("Median Total Sales:", median_sales)
print("Std Dev Total Sales:", std_sales)
print("Max Total Sales:", max_sales)
print("Min Total Sales:", min_sales)
prices_arr = np.array(df["Price"])
print("Prices Array:", prices_arr)
age_min = df["Customer_Age"].min()
age_max = df["Customer_Age"].max()
df["Age_Norm"] = (df["Customer_Age"] - age_min) / (age_max - age_min)
print(df["Age_Norm"].head())

# #1.10
sales_per_month_year = df.groupby(["Year", "Month"])["Sales"].sum().reset_index()
pivot_data = sales_per_month_year.pivot(index="Month", columns="Year", values="Sales")
pivot_data.plot(kind="line", marker="o", figsize=(10,5),color=["white","orange"])
plt.title("Total Sales per Month (2022 vs 2023)")
plt.xlabel("Month")
plt.xticks(range(1, 13))
plt.ylabel("Total Sales")
plt.grid(True)
plt.legend(title="Year")
plt.gca().set_facecolor("#004d4d")
plt.gcf().set_facecolor("#004d4d")
plt.show()
revenue_by_payment = df.groupby("Payment_Method")["Sales"].sum()
revenue_by_payment.plot(kind="pie", autopct="%1.1f%%", startangle=9,colors=["#004d4d", "#006666", "#008080", "#339999", "#66b2b2"])
plt.title("Revenue Share by Payment Method")
plt.ylabel("")
plt.show()
top_products = df.groupby("Product")["Sales"].sum().sort_values(ascending=True).head(5)
top_products.plot(kind="barh", color="#004d4d")
plt.title("Top 5 Products by Sales")
plt.xlabel("Sales")
plt.ylabel("Product")
plt.show()
plt.scatter(df["Price"], df["Quantity"], alpha=0.6, color="#004d4d")
plt.title("Price vs Quantity")
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.show()
sales_category_month = df.groupby(["Month", "Category"])["Sales"].sum().unstack()
sales_category_month.plot(kind="bar", stacked=True,color= ["#004d4d", "#006666", "#008080", "#339999", "#66b2b2"] )
plt.title("Total Sales by Category and Month")
plt.ylabel("Sales")
plt.xlabel("Month")
plt.show()

#1.11

plt.figure(figsize=(8,5))
sns.histplot(df["Price"], bins=50)
plt.title("Distribution of Product Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(x="Category", y="Total_Sales", data=df, palette="Blues")
plt.title("Total Sales by Category")
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(x="Payment_Method", data=df, palette="Blues")
plt.title("Transactions per Payment Method")
plt.show()
plt.figure(figsize=(8,5))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", linewidths=0.5)
plt.title("Correlation Between Numeric Columns")
plt.show()
plt.figure(figsize=(8,5))
sns.barplot(x="Day_of_Week", y="Total_Sales", data=df, estimator=np.mean, palette="Blues")
plt.title("Average Total Sales by Day of the Week")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(8,5))
sns.violinplot(x="Category", y="Quantity", data=df, palette="Blues")
plt.title("Quantity Distribution by Category")
plt.show()
sns.pairplot(df[["Price", "Quantity", "Total_Sales"]], diag_kind="kde", palette="Blues")
plt.suptitle("Pairplot of Price, Quantity, and Total Sales", y=1)
plt.show()


#1.12

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Customer_Age"])
plt.title("Customer Age Before Normalization")
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Age_Norm"])
plt.title("Customer Age After Normalization")
plt.show()

import os

file_path = os.path.abspath("retail_sales_dataset_custom_clean.xlsx")
print("File saved at:", file_path)

df.to_excel(file_path, index=False)




































