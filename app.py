
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

model = joblib.load("sales_model.pkl")
features = joblib.load("features.pkl")
store_df = pd.read_csv("store.csv")

st.set_page_config(layout="wide")
st.title("📊 Retail Sales Forecasting")

store_id = st.selectbox("Select Store ID", store_df["Store"].unique())
promo = st.checkbox("Promo Running?", value=True)
school_holiday = st.checkbox("School Holiday?", value=False)

date_input = st.date_input("Select Date")
year = date_input.year
month = date_input.month
day = date_input.day
day_of_week = date_input.weekday()

store_info = store_df[store_df["Store"] == store_id].iloc[0]
input_data = {
    "Store": store_id,
    "Promo": int(promo),
    "SchoolHoliday": int(school_holiday),
    "Year": year,
    "Month": month,
    "Day": day,
    "DayOfWeek": day_of_week,
    "StoreType": ord(store_info["StoreType"]),
    "Assortment": ord(store_info["Assortment"]),
    "CompetitionDistance": store_info["CompetitionDistance"] or 0,
    "CompetitionOpenSinceMonth": store_info["CompetitionOpenSinceMonth"] or 0,
    "CompetitionOpenSinceYear": store_info["CompetitionOpenSinceYear"] or 0,
    "Promo2": store_info["Promo2"],
    "Promo2SinceWeek": store_info["Promo2SinceWeek"] or 0,
    "Promo2SinceYear": store_info["Promo2SinceYear"] or 0,
}

input_df = pd.DataFrame([input_data])[features]
pred = model.predict(input_df)[0]
st.subheader(f"📈 Predicted Sales: **${pred:,.0f}**")
