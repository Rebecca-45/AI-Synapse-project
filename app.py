import streamlit as st
import pandas as pd
import joblib

st.title("Will AI Usage Improve This Student's GPA?")
st.write("Enter a student's info to predict whether their GPA is likely to improve.")

model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# ---- User inputs ----
weekly_genai_hours = st.slider("Weekly GenAI Hours", 0.0, 40.0, 8.0)
traditional_study_hours = st.slider("Traditional Study Hours", 0.0, 35.0, 10.0)
tool_diversity = st.slider("Tool Diversity (number of tools used)", 1, 5, 2)
perceived_dependency = st.slider("Perceived AI Dependency (1-10)", 1, 10, 5)
anxiety_level = st.slider("Anxiety Level During Exams (1-10)", 1, 10, 5)
major = st.selectbox("Major Category", ["STEM", "Business", "Humanities", "Medical", "Arts"])
year = st.selectbox("Year of Study", ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"])
use_case = st.selectbox("Primary Use Case", [
    "Copywriting/Drafting", "Ideation", "Summarizing_Reading",
    "Debugging/Troubleshooting", "Direct_Answer_Generation"
])
prompt_skill = st.selectbox("Prompt Engineering Skill", ["Beginner", "Intermediate", "Advanced"])

if st.button("Predict"):
    # Build a single-row dataframe matching training format
    input_dict = {
        'Weekly_GenAI_Hours': weekly_genai_hours,
        'Traditional_Study_Hours': traditional_study_hours,
        'Tool_Diversity': tool_diversity,
        'Perceived_AI_Dependency': perceived_dependency,
        'Anxiety_Level_During_Exams': anxiety_level,
        'Major_Category': major,
        'Year_of_Study': year,
        'Primary_Use_Case': use_case,
        'Prompt_Engineering_Skill': prompt_skill,
    }
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)

    # Add any missing columns the model expects, filled with 0
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]  # match column order exactly

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.success(f"Predicted: GPA likely to IMPROVE (confidence: {probability:.0%})")
    else:
        st.error(f"Predicted: GPA likely to NOT improve (confidence: {1-probability:.0%})")