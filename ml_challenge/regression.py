# %% [markdown]
# ## Final Task - Regression Model
# * **Virtual Internship Experience (VIX) Program batch Oktober 2023**
# * **Data Science - Rakamin x Kalbe Nutritionals**
# 
# *Created by: Rakaputra*

# %%
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pma

from sklearn.metrics import mean_squared_error as mse

# %matplotlib inline

# %% [markdown]
# #### **Load/Import Dataset**

# %%
df_customer = pd.read_csv("../dataset/Case Study - Customer.csv", sep=";")
df_product = pd.read_csv("../dataset/Case Study - Product.csv", sep=";")
df_store = pd.read_csv("../dataset/Case Study - Store.csv", sep=";")
df_transaction = pd.read_csv("../dataset/Case Study - Transaction.csv", sep=";")

# %%
df_customer.info()

# %%
df_customer.sample(5)

# %%
df_product.info()

# %%
df_product.sample(5)

# %%
df_store.info()

# %%
df_store.sample(5)

# %%
df_transaction.info()

# %%
df_transaction.sample(5)

# %% [markdown]
# #### **Data Cleaning & Preprocessing**
# * Pengisian nilai kosong pada data Customer

# %%
list(df_customer["Marital Status"].unique())

# %%
df_customer[df_customer["Marital Status"].isna()]

# %%
df_customer.loc[df_customer["Age"] < 30, "Marital Status"].mode()

# %%
df_customer.loc[df_customer["Age"] >= 30, "Marital Status"].mode()

# %%
df_customer.loc[
    df_customer["Marital Status"].isna(), "Marital Status"
] = np.where(df_customer[df_customer["Marital Status"].isna()]["Age"] >= 30,
             "Married", "Single")

df_customer.info()

# %% [markdown]
# Transformasi data "Income" pada dataset "Customer" dan "Date" pada dataset "Transaction"

# %%
df_customer["Income"] = df_customer["Income"].map(lambda x: float(x.replace(",", ".")))
df_transaction["Date"] = pd.to_datetime(df_transaction["Date"], format="%d/%m/%Y")

# %%
df_customer.sample(5)

# %%
df_transaction.sample(5)

# %% [markdown]
# #### **Merge into single Dataframe**

# %%
df_merge = pd.merge(df_transaction, df_customer, on="CustomerID")
df_merge = pd.merge(df_merge, df_product, on="ProductID")
df_merge = pd.merge(df_merge, df_store, on="StoreID")
df_merge.info()

# %%
new_cols_head = ["TransactionID",
                 "Date",
                 "CustomerID",
                 "Age",
                 "Gender",
                 "Marital Status",
                 "Income",
                 "ProductID",
                 "Product Name",
                 "Price_x",
                 "Price_y",
                 "Qty",
                 "TotalAmount",
                 "StoreID",
                 "StoreName",
                 "GroupStore",
                 "Type",
                 "Latitude",
                 "Longitude"]

df_merge = df_merge[new_cols_head]
df_merge.sample(5)

# %% [markdown]
# #### **Data Regression**
# * Memilah data untuk regresi
# * Plotting time series penjualan dalam setahun (sales in a year)

# %%
df_reg = df_merge.groupby("Date")["Qty"].sum().reset_index()
df_reg["Date"] = pd.to_datetime(df_reg["Date"], format="%d/%m/%Y")
df_reg.sort_values(by="Date", inplace=True)
df_reg.set_index("Date", inplace=True)

# %%
df_reg.plot(figsize=(12, 6),
            title="Daily Sales",
            xlabel="Date",
            ylabel="Total Qty",
            legend=False)

# %% [markdown]
# Pembagian data latih (train) dan uji (test)

# %%
df_train = df_reg[:int(0.8 * len(df_reg))]
df_test = df_reg[int(0.8 * len(df_reg)):]

# %% [markdown]
# Pemodelan Arima

# %%
arima_model = pma.auto_arima(
    df_train["Qty"],
    seasonal=False,
    stepwise=False,
    suppress_warnings=True,
    trace=True
)

arima_model.summary()

# %%
p, d, q = arima_model.order
model = SARIMAX(df_train["Qty"].values, order=(p, d, q))
model_fit = model.fit()

# %% [markdown]
# Root Mean Squared Error

# %%
prediction = model_fit.predict(start=len(df_train), end=len(df_train)+len(df_test)-1)
rmse = mse(df_test, prediction, squared=False)
rmse

# %% [markdown]
# #### **Forecasting (Next 90 Days)**

# %%
periods = 90
forecasting = model_fit.forecast(steps=periods)
index = pd.date_range(start="01-01-2023", periods=periods)
df_forecast = pd.DataFrame(forecasting, index=index, columns=["Qty"])
df_forecast.describe()

# %%
plt.figure(figsize=(12, 6))
plt.title("Forecasting Sales")
plt.plot(df_train, label="Train")
plt.plot(df_test, label="Test")
plt.plot(df_forecast, label="Predicted")
plt.legend(loc="best")
plt.show()

# %%
df_forecast.plot(figsize=(12, 6),
                 title="Forecasting Sales",
                 xlabel="Date",
                 ylabel="Total Qty",
                 legend=False)

# %% [markdown]
# Product Forecasting - For next 90 days

# %%
warnings.filterwarnings("ignore")

df_reg_product = df_merge[["Qty", "Date", "Product Name"]]
df_reg_product = df_reg_product.groupby("Product Name")

df_product_forecast = pd.DataFrame({"Date": pd.date_range(start="2023-01-01", periods=90)})

for product_name, data in df_reg_product:
    target_qty = data["Qty"]
    model = SARIMAX(target_qty.values, order=(p, d, q))
    model_fit = model.fit(disp=False)
    forecasting = model_fit.forecast(90)
    df_product_forecast[product_name] = forecasting

df_product_forecast.set_index("Date", inplace=True)
df_product_forecast.sample(5)

# %%
plt.figure(figsize=(10, 6))
for column in df_product_forecast.columns:
    plt.plot(df_product_forecast[column], label=column)

plt.legend(loc="best", bbox_to_anchor=(1, .82))
plt.title("Forecasting Products")
plt.xlabel("Date")
plt.ylabel("Total Qty")
plt.show()
