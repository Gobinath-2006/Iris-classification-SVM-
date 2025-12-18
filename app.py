import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Iris Flower Classification", layout="centered")

st.title("🌸 Iris Flower Classification (SVM)")
st.write("Predict the **Iris flower species** using a trained SVM model")

# Load dataset (optional display)
@st.cache_data
def load_data():
    return pd.read_csv("Iris_dataset(SVM).csv")

df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    with open("Iris Classification(SVM)_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Sidebar inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.number_input(
    "Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1
)
sepal_width = st.sidebar.number_input(
    "Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5
)
petal_length = st.sidebar.number_input(
    "Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4
)
petal_width = st.sidebar.number_input(
    "Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2
)

# Create input dataframe
input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]

    species_map = {
        0: "Iris-setosa 🌼",
        1: "Iris-versicolor 🌺",
        2: "Iris-virginica 🌸"
    }

    st.success(f"### Predicted Species: **{species_map[prediction]}**")

# Dataset preview
with st.expander("📊 View Dataset"):
    st.dataframe(df.head())
