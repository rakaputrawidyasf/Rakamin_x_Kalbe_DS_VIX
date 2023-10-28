# %% [markdown]
# ## Final Task - Clustering Model
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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# %matplotlib inline

# %% [markdown]
# #### **Load/Import Dataset**

# %%
df_customer = pd.read_csv("../dataset/Case Study - Customer.csv", sep=";")
df_transaction = pd.read_csv("../dataset/Case Study - Transaction.csv", sep=";")

# %%
df_customer.info()

# %%
df_customer.sample(5)

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
df_merge.info()

# %%
new_cols_head = ["Date",
                 "TransactionID",
                 "CustomerID",
                 "Age",
                 "Gender",
                 "Marital Status",
                 "Income",
                 "ProductID",
                 "Price",
                 "Qty",
                 "TotalAmount",
                 "StoreID"]

df_merge = df_merge[new_cols_head]
df_merge.sample(5)

# %% [markdown]
# #### **Data Clustering**
# * Agregasi data

# %%
aggregation = {
    "TransactionID": "count",
    "Qty": "sum",
    "TotalAmount": "sum",
}

df_cluster = df_merge.groupby("CustomerID").aggregate(aggregation).reset_index()
df_cluster.info()

# %%
df_cluster.sample(5)

# %% [markdown]
# Penyeragaman skala data

# %%
scaler = StandardScaler()
df_scaler = scaler.fit_transform(df_cluster[["TransactionID", "Qty", "TotalAmount"]])
df_scaler = pd.DataFrame(df_scaler, columns=["TransactionID", "Qty", "TotalAmount"])
df_scaler.sample(5)

# %% [markdown]
# Menentukan n cluster optimal

# %%
warnings.filterwarnings("ignore")

inertia = []
max_clusters = 12
for cluster in range(1, max_clusters):
    kmeans = KMeans(n_clusters=cluster, random_state=42, n_init=cluster)
    kmeans.fit(df_cluster.drop("CustomerID", axis=1))
    inertia.append(kmeans.inertia_)

# %%
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, max_clusters), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(np.arange(1, max_clusters))
plt.show()

# %% [markdown]
# Clustering

# %%
df_cluster.drop("CustomerID", axis=1, inplace=True)
clusters = 3
kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=clusters)
kmeans.fit(df_cluster)
df_cluster["cluster"] = kmeans.labels_

# %%
warnings.filterwarnings("ignore")

plt.figure(figsize=(8, 8))
sn.pairplot(data=df_cluster, hue="cluster", palette="bright")
plt.show()
