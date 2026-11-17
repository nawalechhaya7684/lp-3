
# Uber Fare Prediction: Linear Regression vs Random Forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------- 1. Read and pre-process ----------
df = pd.read_csv("uber.csv")
print(df.head())

df = df.drop(["Unnamed: 0", "key"], axis=1, errors="ignore")   # useless columns
df = df.dropna()                                                # remove missing values

# split date into useful parts
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df = df.drop("pickup_datetime", axis=1)

# distance (km) between pickup and drop using Haversine formula
def distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

df["distance"] = distance(df["pickup_latitude"], df["pickup_longitude"],
                          df["dropoff_latitude"], df["dropoff_longitude"])

# ---------- 2. Outliers ----------
sns.boxplot(x=df["fare_amount"])
plt.title("Fare amount (before removing outliers)")
plt.show()

def remove_outliers(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(col, "outliers:", ((data[col] < low) | (data[col] > high)).sum())
    return data[(data[col] >= low) & (data[col] <= high)]

df = df[(df["fare_amount"] > 0) & (df["passenger_count"] > 0)]   # impossible values
df = remove_outliers(df, "fare_amount")
df = remove_outliers(df, "distance")
print("Rows left:", len(df))

# ---------- 3. Correlation ----------
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation")
plt.show()

# ---------- 4. Models ----------
X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=50, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ---------- 5. Evaluation ----------
def show(name, pred):
    print("\n", name)
    print("R2   :", r2_score(y_test, pred))
    print("RMSE :", np.sqrt(mean_squared_error(y_test, pred)))
    print("MAE  :", mean_absolute_error(y_test, pred))

show("Linear Regression", lr_pred)
show("Random Forest", rf_pred)

# ---------- 6. Predict the fare ----------
plat = float(input("\nPickup latitude: "))
plon = float(input("Pickup longitude: "))
dlat = float(input("Drop latitude: "))
dlon = float(input("Drop longitude: "))
people = int(input("Passengers: "))

new_ride = pd.DataFrame([{
    "pickup_longitude": plon, "pickup_latitude": plat,
    "dropoff_longitude": dlon, "dropoff_latitude": dlat,
    "passenger_count": people,
    "hour": 10, "day": 15, "month": 6, "year": 2015,      # assumed date and time
    "distance": distance(plat, plon, dlat, dlon)}])
new_ride = new_ride[X.columns]                             # same column order as training

print("Predicted fare ($):", rf.predict(new_ride)[0])


# -----------------------------------------
# SAMPLE INPUT:
# Dataset: uber.csv
# Pickup latitude: 40.7580
# Pickup longitude: -73.9855
# Dropoff latitude: 40.7484
# Dropoff longitude: -73.9857
# Number of passengers: 2
#
# SAMPLE OUTPUT:
#Predicted fare ($): 8.6
#
# INSTALL:
# python -m pip install pandas numpy scikit-learn
#
# RUN:
# python P6.py
# -----------------------------------------
