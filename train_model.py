import pandas as pd
import xgboost as xgb
import joblib

train = pd.read_csv("train.csv", parse_dates=["Date"])
store = pd.read_csv("store.csv")
df = pd.merge(train, store, on="Store", how="left")
df = df[df["Open"] == 1]

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

drop_cols = ["Date", "Customers", "Open", "StateHoliday"]
df = df.drop(columns=drop_cols)
df = df.fillna(0)

X = df.drop(columns=["Sales"])
y = df["Sales"]

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X, y)

joblib.dump(model, "sales_model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
