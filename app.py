import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Performance Prediction")
st.subheader("Machine Learning Based Prediction System")

st.write(
    "Enter the student's details below to predict the final score."
)

# -----------------------------
# Dataset
# -----------------------------
data = {
    "study_hours": [
        2, 3, 4, 5, 6, 7, 1, 8, 4, 5,
        3, 6, 2, 7, 5, 9, 2, 4, 6, 3
    ],

    "attendance": [
        65, 70, 75, 80, 85, 90, 60, 95, 78, 82,
        72, 88, 68, 92, 79, 96, 63, 76, 87, 69
    ],

    "previous_score": [
        55, 60, 65, 70, 75, 82, 50, 88, 68, 74,
        62, 80, 58, 85, 72, 92, 54, 67, 78, 61
    ],

    "assignment_score": [
        60, 65, 70, 75, 80, 85, 55, 90, 72, 78,
        68, 84, 62, 88, 76, 94, 58, 71, 82, 64
    ],

    "sleep_hours": [
        6, 7, 7, 8, 7, 8, 6, 8, 6, 7,
        8, 7, 6, 8, 7, 8, 6, 7, 8, 6
    ],

    "extracurricular": [
        0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
        0, 1, 1, 1, 0, 1, 0, 1, 0, 1
    ],

    "final_score": [
        58, 64, 69, 76, 82, 89, 52, 94, 71, 79,
        66, 85, 60, 92, 77, 97, 56, 70, 84, 63
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Prepare Machine Learning Data
# -----------------------------
X = df.drop("final_score", axis=1)
y = df["final_score"]

# -----------------------------
# Train Random Forest Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

model.fit(X, y)

# -----------------------------
# Student Input Section
# -----------------------------
st.markdown("### 📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=15.0,
        value=6.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

    previous_score = st.number_input(
        "Previous Score",
        min_value=0.0,
        max_value=100.0,
        value=78.0,
        step=1.0
    )

with col2:

    assignment_score = st.number_input(
        "Assignment Score",
        min_value=0.0,
        max_value=100.0,
        value=82.0,
        step=1.0
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=15.0,
        value=7.0,
        step=0.5
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["No", "Yes"]
    )

# -----------------------------
# Prediction Button
# -----------------------------
if st.button(
    "🔮 Predict Final Score",
    use_container_width=True
):

    # Convert Yes/No to 1/0
    extracurricular_value = (
        1 if extracurricular == "Yes" else 0
    )

    # Create input DataFrame
    new_student = pd.DataFrame({

        "study_hours": [study_hours],

        "attendance": [attendance],

        "previous_score": [previous_score],

        "assignment_score": [assignment_score],

        "sleep_hours": [sleep_hours],

        "extracurricular": [extracurricular_value]
    })

    # Make prediction
    prediction = model.predict(new_student)[0]

    # -----------------------------
    # Display Prediction
    # -----------------------------
    st.success(
        f"🎯 Predicted Final Score: {prediction:.2f} / 100"
    )

    # Performance category
    if prediction >= 75:

        st.info(
            "📈 Performance Category: Good"
        )

    elif prediction >= 50:

        st.warning(
            "📊 Performance Category: Average"
        )

    else:

        st.error(
            "⚠️ Performance Category: Needs Improvement"
        )

    # -----------------------------
    # Display Student Information
    # -----------------------------
    st.markdown("### 📋 Student Information")

    display_df = new_student.copy()

    display_df["extracurricular"] = (
        display_df["extracurricular"]
        .map({0: "No", 1: "Yes"})
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )

# -----------------------------
# About Project
# -----------------------------
with st.expander("ℹ️ About This Project"):

    st.write(
        "This project uses Machine Learning to predict "
        "a student's final score based on academic and "
        "personal-performance features."
    )

    st.write(
        "Model used: Random Forest Regression"
    )

    st.write(
        "Input features: Study Hours, Attendance, "
        "Previous Score, Assignment Score, Sleep Hours "
        "and Extracurricular Activities."
    )

    st.write(
        "The dataset contains 20 synthetic records "
        "created for educational demonstration."
    )

    st.warning(
        "The predicted score is an estimate and is not "
        "a guaranteed actual score."
    )

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Student Performance Prediction | "
    "Python + Pandas + Scikit-learn + Streamlit"
)