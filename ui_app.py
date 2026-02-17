import streamlit as st
import requests

st.title("Farmer Friendly ET Calculator")

Tmax = st.number_input("Enter Tmax (°C)", value=30.0)
Tmin = st.number_input("Enter Tmin (°C)", value=18.0)

RHmax = st.number_input("RHmax (optional)", value=0.0)
RHmin = st.number_input("RHmin (optional)", value=0.0)
n = st.number_input("Sunshine hours n (optional)", value=0.0)
u2 = st.number_input("Wind speed u2 (optional)", value=0.0)

if st.button("Calculate ET"):

    data = {"Tmax":Tmax,"Tmin":Tmin}

    if RHmax>0: data["RHmax"]=RHmax
    if RHmin>0: data["RHmin"]=RHmin
    if n>0: data["n"]=n
    if u2>0: data["u2"]=u2

    res = requests.post("http://127.0.0.1:5000/predict",json=data)

    if res.status_code==200:
        out = res.json()
        st.success(f"Scenario used: {out['Scenario']}")
        st.success(f"Predicted ET: {out['ET']:.2f} mm/day")
    else:
        st.error("Backend not responding")
