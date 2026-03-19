import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pickle
import requests

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
# CUSTOM CSS (Professional UI)
# -------------------------

st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
background: linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.metric-card{
background-color: rgba(255,255,255,0.1);
padding:20px;
border-radius:15px;
box-shadow:0px 4px 10px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# st.markdown('<p class="main-title">🌍 India Air Pollution AI Dashboard</p>', unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------

model = pickle.load(open("models/aqi_model.pkl","rb"))

# -------------------------
# API KEY
# -------------------------

API_KEY = "b08c7b2d4f64b8d7a9deb494852fa66e"

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

selected_city = st.sidebar.selectbox(
    "Select City (Search Enabled)",
    list(cities.keys())
)

pm25 = st.sidebar.number_input("PM2.5",0.0,500.0,50.0,step=1.0)
pm10 = st.sidebar.number_input("PM10",0.0,500.0,80.0,step=1.0)
no2 = st.sidebar.number_input("NO2",0.0,200.0,30.0,step=1.0)
co = st.sidebar.number_input("CO",0.0,10.0,1.0,step=0.1)
o3 = st.sidebar.number_input("O3",0.0,200.0,20.0,step=1.0)

predict = st.sidebar.button("Predict AQI")

# -------------------------
# PREDICTION
# -------------------------

if predict:

    input_data = np.array([[pm25,pm10,no2,co,o3]])
    st.session_state.prediction = model.predict(input_data)[0]

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

    # Gauge meter
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

    aqi_levels = {
    1:"Good 🟢",
    2:"Fair 🟡",
    3:"Moderate 🟠",
    4:"Poor 🔴",
    5:"Very Poor 🟣"
    }

    st.info(f"Live AQI for {selected_city}: {aqi_levels[live_aqi]}")

    st.write("### Pollutant Values")

    c1,c2,c3 = st.columns(3)

    c1.metric("PM2.5",f"{comp['pm2_5']} µg/m³")
    c2.metric("PM10",f"{comp['pm10']} µg/m³")
    c3.metric("NO2",f"{comp['no2']} µg/m³")

    c4,c5 = st.columns(2)

    c4.metric("CO",f"{comp['co']} µg/m³")
    c5.metric("O3",f"{comp['o3']} µg/m³")

except:

    st.warning("Unable to fetch live AQI data")

# -------------------------
# POLLUTION CONTRIBUTION
# -------------------------

st.subheader("Pollution Contribution")

pollution = pd.DataFrame({
"Pollutant":["PM2.5","PM10","NO2","CO","O3"],
"Level":[pm25,pm10,no2,co,o3]
})

fig = px.bar(
    pollution,
    x="Pollutant",
    y="Level",
    color="Level",
    color_continuous_scale="Turbo"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------
# MULTI-CITY MAP
# -------------------------

st.subheader("India Air Pollution Map")

india_map = folium.Map(location=[22.5937,78.9629],zoom_start=5)

city_aqi_demo = {
"Delhi":251,
"Mumbai":142,
"Kolkata":142,
"Chennai":110,
"Bengaluru":97,
"Hyderabad":112,
"Ahmedabad":238,
"Lucknow":212,
"Patna":220,
"Bhubaneswar":161
}

def get_color(aqi):

    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "yellow"
    elif aqi <= 200:
        return "orange"
    elif aqi <= 300:
        return "red"
    elif aqi <= 400:
        return "purple"
    else:
        return "black"

for city in cities:

    lat,lon = cities[city]
    aqi = city_aqi_demo[city]

    folium.CircleMarker(
        location=[lat,lon],
        radius=12,
        popup=f"{city} AQI: {aqi}",
        color=get_color(aqi),
        fill=True,
        fill_color=get_color(aqi)
    ).add_to(india_map)

st_folium(india_map,width=1200)

# -------------------------
# AI AQI FORECAST (7 DAYS)
# -------------------------

st.subheader("AI Air Pollution Forecast (Next 7 Days)")

days = ["Day 1","Day 2","Day 3","Day 4","Day 5","Day 6","Day 7"]

future_pm25 = [pm25 + np.random.uniform(-5,10) for _ in range(7)]
future_pm10 = [pm10 + np.random.uniform(-5,10) for _ in range(7)]
future_no2 = [no2 + np.random.uniform(-3,8) for _ in range(7)]
future_co = [co + np.random.uniform(-0.2,0.5) for _ in range(7)]
future_o3 = [o3 + np.random.uniform(-2,6) for _ in range(7)]

forecast_aqi = []

for i in range(7):

    input_data = pd.DataFrame([{
        "PM2.5": future_pm25[i],
        "PM10": future_pm10[i],
        "NO2": future_no2[i],
        "CO": future_co[i],
        "O3": future_o3[i]
    }])

    pred = model.predict(input_data)[0]
    forecast_aqi.append(pred)

forecast_df = pd.DataFrame({
"Day":days,
"AQI":forecast_aqi
})

# Chart
fig = px.line(
    forecast_df,
    x="Day",
    y="AQI",
    markers=True,
    title="Predicted AQI Trend (Next 7 Days)"
)

st.plotly_chart(fig,use_container_width=True)

st.dataframe(forecast_df)