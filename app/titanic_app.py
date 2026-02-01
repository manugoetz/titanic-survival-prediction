import streamlit as st
import pandas as pd
import joblib
import sys
import os
import Path
st.write("CWD:", os.getcwd())
st.write("Files:", os.listdir("."))
# Add the parent folder to sys.path
# sys.path.append(os.path.abspath(os.path.join('..')))
# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

st.write("CWD:", os.getcwd())
st.write("Files:", os.listdir("."))
import src.feature_engineering as fen

# Load model
@st.cache_resource
def load_model():
    return joblib.load("../models/pipeline.pkl")

model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability.")

TITLE_RULES = {
    "male": {
        "child": ["Master"],
        "adult": [
            "Mr", "Dr", "Rev",
            "Capt", "Col", "Major",
            "Sir", "Don", "Jonkheer"
        ]
    },
    "female": {
        "child": ["Miss"],
        "adult": [
            "Miss", "Mlle", "Ms",
            "Mrs", "Mme", "Dr",
            "Lady", "the Countess", "Dona"
        ]
    }
}

def get_available_titles(sex, age):
    if age < 14:
        return TITLE_RULES[sex]["child"]
    else:
        return TITLE_RULES[sex]["adult"]


# --- User inputs ---
sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 30)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
sibsp = st.number_input("Number of Siblings + Spouses on Board", 0, 8, 0)
parch = st.number_input("Number of Parents + Children on Board", 0, 6, 0)
embarked = st.selectbox("Embarked", ["Southampton (ENG)", "Cherbourg (FRA)", "Queenstown (IRL)"])
available_titles = get_available_titles(sex, age)
title = st.selectbox(
    "Title",
    available_titles
)
cabin = st.text_input("Cabin (optional)", "")
ticket = st.text_input("Ticket", "A/5 21171")

# Create DataFrame
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "Fare": fare,
    "SibSp": sibsp,
    "Parch": parch,
    "Embarked": embarked[0], # only first letter
    "Title": title,
    "Cabin": cabin if cabin else None,
    "Ticket": ticket
}])

# Feature engineering
# input_df = fen.extract_title(input_df)
input_df = fen.get_famtype(input_df)
input_df = fen.map_ticket(input_df, threshold=3)
input_df = fen.bin_fare(input_df)
input_df = fen.extract_deck(input_df)

# Predict
if st.button("Predict Survival"):
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Result")
    st.write(f"🧮 Survival Probability: **{proba:.2%}**")

    if pred == 1:
        st.success("🎉 Likely to Survive")
    else:
        st.error("⚠️ Unlikely to Survive")
