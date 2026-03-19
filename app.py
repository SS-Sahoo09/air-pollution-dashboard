import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pickle
import requests
import os

# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(
    page_title="India Air Pollution AI Dashboard",
    layout="wide",
    page_icon="🌍"
)

st.markdown("# 🌍 India Air Pollution AI Dashboard")
st.caption("Machine Learning Based Air Quality Monitoring & Prediction System")

# -------------------------
# SAFE MODEL LOADING (FIXED)
# -------------------------

model = None

if os.path.exists("models/aqi_model.pkl"):
    try:
        model = pickle.load(open("models/aqi_model.pkl", "rb"))
    except:
        model = None

if model is None:
    st.warning("⚠️ Model not found — using demo prediction")

# -------------------------
# API KEY
# -------------------------

API_KEY = "YOUR_API_KEY_HERE"

# -------------------------
# CITY DATA
# -------------------------

cities = {
"Delhi":[28.7041,77.1025],
"Mumbai":[19.0760,72.8777],
"Kolkata":[22.5726,88.3639],
"Chennai":[13.0827,80.2707],
"Bengaluru":[12.9716,77.5946],
"Hyderabad":[17.3850,78.4867],
"Ahmedabad":[23.0225,72.5714],
"Lucknow":[26.8467,80.9462],
"Patna":[25.5941,85.1376],
"Bhubaneswar":[20.2961,85.8245]
}

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.header("AQI Prediction")

selected_city = st.sidebar.selectbox("Select City", list(cities.keys()))

pm25 = st.sidebar.number_input("PM2.5",0.0,500.0,50.0)
pm10 = st.sidebar.number_input("PM10",0.0,500.0,80.0)
no2 = st.sidebar.number_input("NO2",0.0,200.0,30.0)
co = st.sidebar.number_input("CO",0.0,10.0,1.0)
o3 = st.sidebar.number_input("O3",0.0,200.0,20.0)

predict = st.sidebar.button("Predict AQI")

# -------------------------
# PREDICTION (FIXED)
# -------------------------

if predict:
    input_data = np.array([[pm25,pm10,no2,co,o3]])

    if model is not None:
        prediction = model.predict(input_data)[0]
    else:
        prediction = np.mean(input_data)

    st.session_state.prediction = prediction

# -------------------------
# KPI CARDS
# -------------------------

st.subheader("Air Pollution Statistics")

col1,col2,col3 = st.columns(3)
col1.metric("Average AQI India","155")
col2.metric("Most Polluted City","Delhi")
col3.metric("Cleanest State","Mizoram")

# -------------------------
# SHOW PREDICTION
# -------------------------

if "prediction" in st.session_state:

    prediction = st.session_state.prediction

    st.subheader("Prediction Result")
    st.success(f"Predicted AQI : {prediction:.2f}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text':"AQI Level"},
        gauge={
            'axis':{'range':[0,500]},
            'steps':[
                {'range':[0,50],'color':'green'},
                {'range':[50,100],'color':'yellow'},
                {'range':[100,200],'color':'orange'},
                {'range':[200,300],'color':'red'},
                {'range':[300,400],'color':'purple'},
                {'range':[400,500],'color':'black'}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

# -------------------------
# LIVE AQI DATA
# -------------------------

st.subheader(f"Live AQI Data — {selected_city}")

lat,lon = cities[selected_city]

url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

try:
    response = requests.get(url).json()

    live_aqi = response["list"][0]["main"]["aqi"]
    comp = response["list"][0]["components"]

    levels = {1:"Good 🟢",2:"Fair 🟡",3:"Moderate 🟠",4:"Poor 🔴",5:"Very Poor 🟣"}

    st.info(f"Live AQI: {levels[live_aqi]}")

    c1,c2,c3 = st.columns(3)
    c1.metric("PM2.5",comp['pm2_5'])
    c2.metric("PM10",comp['pm10'])
    c3.metric("NO2",comp['no2'])

except:
    st.warning("API Error")

# -------------------------
# POLLUTION CHART
# -------------------------

st.subheader("Pollution Contribution")

pollution = pd.DataFrame({
"Pollutant":["PM2.5","PM10","NO2","CO","O3"],
"Level":[pm25,pm10,no2,co,o3]
})

fig = px.bar(pollution,x="Pollutant",y="Level",color="Level")
st.plotly_chart(fig,use_container_width=True)

# -------------------------
# MAP
# -------------------------

st.subheader("India Map")

m = folium.Map(location=[22,78],zoom_start=5)

for city in cities:
    lat,lon = cities[city]
    folium.CircleMarker([lat,lon],radius=10,color="red",fill=True).add_to(m)

st_folium(m,width=1200)

# -------------------------
# FORECAST (FIXED)
# -------------------------

st.subheader("7-Day AQI Forecast")

days = ["D1","D2","D3","D4","D5","D6","D7"]
forecast = []

for i in range(7):

    sample = np.array([[pm25,pm10,no2,co,o3]])

    if model is not None:
        val = model.predict(sample)[0]
    else:
        val = np.mean(sample)

    forecast.append(val + np.random.uniform(-10,10))

df = pd.DataFrame({"Day":days,"AQI":forecast})

fig = px.line(df,x="Day",y="AQI",markers=True)
st.plotly_chart(fig,use_container_width=True)

st.dataframe(df)
