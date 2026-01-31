import streamlit as st
# import pandas as pd
# import joblib

# from features.feature_engineering import (
#     extract_title, extract_deck, get_famtype, bin_fare
# )

# # Load model
# @st.cache_resource
# def load_model():
#     return joblib.load("model/pipeline.pkl")

# model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability.")

# # --- User inputs ---
# sex = st.selectbox("Sex", ["male", "female"])
# pclass = st.selectbox("Passenger Class", [1, 2, 3])
# age = st.slider("Age", 0, 80, 30)
# fare = st.number_input("Fare", min_value=0.0, value=30.0)
# sibsp = st.number_input("Siblings / Spouses", 0, 8, 0)
# parch = st.number_input("Parents / Children", 0, 6, 0)
# embarked = st.selectbox("Embarked", ["S", "C", "Q"])
# name = st.text_input("Name", "Doe, Mr. John")
# cabin = st.text_input("Cabin (optional)", "")
# ticket = st.text_input("Ticket", "A/5 21171")

# # Create DataFrame
# input_df = pd.DataFrame([{
#     "Pclass": pclass,
#     "Sex": sex,
#     "Age": age,
#     "Fare": fare,
#     "SibSp": sibsp,
#     "Parch": parch,
#     "Embarked": embarked,
#     "Name": name,
#     "Cabin": cabin if cabin else None,
#     "Ticket": ticket
# }])

# # Feature engineering
# input_df = extract_title(input_df)
# input_df = extract_deck(input_df)
# input_df = get_famtype(input_df)
# input_df = bin_fare(input_df)

# # Predict
# if st.button("Predict Survival"):
#     proba = model.predict_proba(input_df)[0][1]
#     pred = model.predict(input_df)[0]

#     st.subheader("Result")
#     st.write(f"🧮 Survival Probability: **{proba:.2%}**")

#     if pred == 1:
#         st.success("🎉 Likely to Survive")
#     else:
#         st.error("⚠️ Unlikely to Survive")
