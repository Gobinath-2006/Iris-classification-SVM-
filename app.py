import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Iris SVM App", layout="centered")

st.title("🌸 Iris Flower Classification (SVM)")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    with open("Iris Classification(SVM)_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------- Load Dataset (SAFE) ----------
def load_data():
    file_name = "Iris_dataset(SVM).csv"
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    else:
        return None

df = load_data()

# ---------- Sidebar Inputs ----------
st.sidebar.header("Input Features")

sepal_length = st.sidebar.number_input("Sepal Length", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.number_input("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.sidebar.number_input("Petal Length", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.number_input("Petal Width", 0.1, 2.5, 0.2)

input_df = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

# ---------- Prediction ----------
if st.button("🔍 Predict"):
    pred = model.predict(input_df)[0]

    species = {
        0: "Iris-setosa 🌼",
        1: "Iris-versicolor 🌺",
        2: "Iris-virginica 🌸"
    }

    st.success(f"### Predicted Species: {species[pred]}")

# ---------- Dataset Preview ----------
st.subheader("📊 Dataset Preview")

if df is not None:
    st.dataframe(df.head())
else:
    st.warning("Dataset file not found. App is running using the trained model only.")
