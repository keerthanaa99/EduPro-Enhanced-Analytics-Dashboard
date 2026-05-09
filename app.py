import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="EduPro Analytics Dashboard", layout="wide")

st.title("🎓 EduPro Course Demand Prediction Dashboard")
st.markdown("Predict course enrollments using Machine Learning")

# ===============================
# SETTINGS
# ===============================
CHART_HEIGHT = 450
TOP_N = 6   # reduced for cleaner UI

# ===============================
# LOAD DATA
# ===============================
file_path = "EduPro Online Platform.xlsx"

courses = pd.read_excel(file_path, sheet_name="Courses")
teachers = pd.read_excel(file_path, sheet_name="Teachers")
transactions = pd.read_excel(file_path, sheet_name="Transactions")

data = pd.merge(transactions, courses, on="CourseID", how="left")
data = pd.merge(data, teachers, on="TeacherID", how="left")

course_data = data.groupby('CourseID').agg({
    'Amount': 'sum',
    'TransactionID': 'count',
    'CourseDuration': 'mean',
    'CourseRating': 'mean',
    'YearsOfExperience': 'mean',
    'TeacherRating': 'mean'
}).reset_index()

course_data.rename(columns={
    'TransactionID': 'EnrollmentCount',
    'Amount': 'TotalRevenue'
}, inplace=True)

course_data = pd.merge(course_data, courses, on='CourseID', how='left')

course_data.rename(columns={
    'CourseDuration_x': 'CourseDuration',
    'CourseRating_x': 'CourseRating'
}, inplace=True)

# ===============================
# MODEL
# ===============================
features = [
    'CourseDuration',
    'CourseRating',
    'YearsOfExperience',
    'TeacherRating',
    'CourseCategory',
    'CourseLevel'
]

X = pd.get_dummies(course_data[features], drop_first=True)
y = course_data['EnrollmentCount']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("📥 Enter Course Details")

duration = st.sidebar.slider("Course Duration (hours)", 1, 60, 20)
rating = st.sidebar.slider("Course Rating", 1.0, 5.0, 4.0)
experience = st.sidebar.slider("Instructor Experience (years)", 1, 30, 5)
teacher_rating = st.sidebar.slider("Teacher Rating", 1.0, 5.0, 4.0)

category = st.sidebar.selectbox("Course Category", course_data['CourseCategory'].dropna().unique())
level = st.sidebar.selectbox("Course Level", course_data['CourseLevel'].dropna().unique())

st.sidebar.markdown("---")
analyze = st.sidebar.button("🚀 Analyze Course", use_container_width=True)

# ===============================
# MAIN LOGIC
# ===============================
if not analyze:
    st.info("👉 Enter details and click **Analyze Course**")
else:
    input_df = pd.DataFrame([{
        'CourseDuration': duration,
        'CourseRating': rating,
        'YearsOfExperience': experience,
        'TeacherRating': teacher_rating,
        'CourseCategory': category,
        'CourseLevel': level
    }])

    input_encoded = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    # ===============================
    # METRICS
    # ===============================
    st.subheader("📊 Prediction Result")
    c1, c2, c3 = st.columns(3)

    c1.metric("📈 Predicted Enrollments", int(prediction))
    c2.metric("⭐ Course Rating", rating)
    c3.metric("👨‍🏫 Experience", experience)

    st.markdown("---")

    # ===============================
    # PREDICTION vs AVERAGE
    # ===============================
    st.subheader("📈 Prediction vs Average")

    avg = course_data['EnrollmentCount'].mean()

    df_compare = pd.DataFrame({
        "Type": ["Predicted", "Average"],
        "Enrollments": [prediction, avg]
    })

    fig1 = px.bar(
        df_compare,
        x="Type",
        y="Enrollments",
        text="Enrollments",
        color="Type",
        template="plotly_dark"
    )

    fig1.update_traces(texttemplate='%{text:.0f}', textposition='outside')

    st.plotly_chart(fig1, use_container_width=True)

    # ===============================
    # TOP COURSES (GLOBAL)
    # ===============================
    st.subheader(f"🏆 Top {TOP_N} Performing Courses")

    top_courses = course_data.nlargest(TOP_N, 'EnrollmentCount').copy()
    top_courses['CourseID'] = top_courses['CourseID'].astype(str)

    fig2 = px.bar(
        top_courses,
        x='CourseID',
        y='EnrollmentCount',
        text='EnrollmentCount',
        color='EnrollmentCount',
        template="plotly_dark"
    )

    fig2.update_traces(texttemplate='%{text:.0f}', textposition='outside')

    st.plotly_chart(fig2, use_container_width=True)

    # ===============================
    # FEATURE IMPORTANCE (FIXED LABELS)
    # ===============================
    st.subheader("🔍 Feature Importance")

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False).head(TOP_N)

    importance.index = (
        importance.index
        .str.replace("_", " ")
        .str.replace("Coursecategory", "Category")
        .str.replace("Courselevel", "Level")
        .str.title()
    )

    imp_df = importance.reset_index()
    imp_df.columns = ["Feature", "Importance"]

    fig3 = px.bar(
        imp_df,
        x="Feature",
        y="Importance",
        text="Importance",
        color="Importance",
        template="plotly_dark"
    )

    fig3.update_layout(xaxis_tickangle=-30)
    fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside')

    st.plotly_chart(fig3, use_container_width=True)

    # ===============================
    # CATEGORY FILTERED COURSES (FIXED)
    # ===============================
    st.subheader(f"📚 Top Courses in {category} ({level})")

    filtered = course_data[
        (course_data['CourseCategory'] == category) &
        (course_data['CourseLevel'] == level)
    ]

    if filtered.empty:
        st.warning("No data available")
    else:
        filtered = filtered.nlargest(TOP_N, 'EnrollmentCount')
        filtered['CourseID'] = filtered['CourseID'].astype(str)

        fig4 = px.bar(
            filtered,
            x='CourseID',
            y='EnrollmentCount',
            text='EnrollmentCount',
            color='EnrollmentCount',
            template="plotly_dark"
        )

        fig4.update_traces(texttemplate='%{text:.0f}', textposition='outside')

        st.plotly_chart(fig4, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("💡 Developed for EduPro Predictive Analytics Project")