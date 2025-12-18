import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.set_page_config(page_title="Iris SVM App", layout="centered")

st.title("🌸 Iris Flower Classification (SVM)")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    with open("Iris Classification(SVM)_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------- Sidebar Inputs ----------
st.sidebar.header("Input Features")

sepal_length = st.sidebar.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

# ---------- Convert to NumPy (IMPORTANT FIX) ----------
input_data = np.array([
    [sepal_length, sepal_width, petal_length, petal_width]
])

# ---------- Prediction ----------
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]

    species = {
        0: "Iris-setosa 🌼",
        1: "Iris-versicolor 🌺",
        2: "Iris-virginica 🌸"
    }

    st.success(f"### Predicted Species: {species[prediction]}")
