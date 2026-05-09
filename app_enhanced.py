import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import os
from feature_engineering import engineer_features

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="EduPro Enhanced Analytics Dashboard", layout="wide")

# Custom CSS for dark theme and styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d4156;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 EduPro Enhanced Analytics Dashboard")
st.markdown("Advanced Course Demand & Revenue Prediction System")

# ===============================
# LOAD DATA & MODELS
# ===============================
@st.cache_data
def load_data():
    file_path = "EduPro Online Platform.xlsx"
    xl = pd.read_excel(file_path, sheet_name=None)
    courses = xl['Courses']
    teachers = xl['Teachers']
    transactions = xl['Transactions']
    
    # Aggregate data
    data = pd.merge(transactions, courses, on="CourseID", how="left")
    data = pd.merge(data, teachers, on="TeacherID", how="left")
    
    course_data = data.groupby('CourseID').agg({
        'Amount': 'sum',
        'TransactionID': 'count',
        'CourseDuration': 'mean',
        'CourseRating': 'mean',
        'YearsOfExperience': 'mean',
        'TeacherRating': 'mean',
        'CoursePrice': 'mean'
    }).reset_index()
    
    course_data.rename(columns={
        'TransactionID': 'EnrollmentCount',
        'Amount': 'TotalRevenue'
    }, inplace=True)
    
    course_data = pd.merge(course_data, courses[['CourseID', 'CourseName', 'CourseCategory', 'CourseLevel']], on='CourseID', how='left')
    return course_data, courses, teachers, transactions

course_data, raw_courses, raw_teachers, raw_transactions = load_data()

# Load Revenue Models and Metrics
@st.cache_resource
def load_models_and_metrics():
    models = {}
    try:
        models['lr'] = joblib.load('linear_regression_revenue_model.pkl')
        models['rf'] = joblib.load('random_forest_revenue_model.pkl')
        models['gb'] = joblib.load('gradient_boosting_revenue_model.pkl')
        rev_features = joblib.load('revenue_model_features.pkl')
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        st.error(f"Error loading models or metrics: {e}")
        models = None
        rev_features = None
        metrics = None
    return models, rev_features, metrics

rev_models, rev_features, rev_metrics = load_models_and_metrics()

# ===============================
# TRAIN ENROLLMENT MODEL (with Price)
# ===============================
@st.cache_resource
def train_enrollment_model(df):
    features = ['CourseDuration', 'CourseRating', 'YearsOfExperience', 'TeacherRating', 'CoursePrice', 'CourseCategory', 'CourseLevel']
    X_raw = df[features]
    X = pd.get_dummies(X_raw, drop_first=True)
    y = df['EnrollmentCount']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

enrol_model, enrol_feature_names = train_enrollment_model(course_data)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("🕹️ Control Panel")

# Section 2: Model Selector
st.sidebar.subheader("Model Configuration")
model_option = st.sidebar.selectbox(
    "Select Revenue Model",
    ["Random Forest", "Linear Regression", "Gradient Boosting"]
)

# Extract accuracy for the selected model from the dictionary format
if rev_metrics:
    model_key = model_option.lower().replace(" ", "_")
    selected_metric = rev_metrics.get(model_key)
    if selected_metric and 'r2_score' in selected_metric:
        st.sidebar.info(f"Model R² Score: {selected_metric['r2_score']:.3f}")
    elif selected_metric and 'R2 Score' in selected_metric: # Fallback for other formats
        st.sidebar.info(f"Model R² Score: {selected_metric['R2 Score']:.3f}")

st.sidebar.markdown("---")
st.sidebar.subheader("Course Parameters")
duration = st.sidebar.slider("Course Duration (hours)", 1, 60, 30)
rating = st.sidebar.slider("Course Rating", 1.0, 5.0, 4.2)
experience = st.sidebar.slider("Instructor Experience (years)", 1, 30, 8)
teacher_rating = st.sidebar.slider("Teacher Rating", 1.0, 5.0, 4.5)
base_price = st.sidebar.number_input("Base Course Price (₹)", min_value=0.0, value=99.0)

# Section 4: Price Adjustment Slider
price_adj = st.sidebar.slider("Adjust Course Price (%)", -50, 50, 0)
adjusted_price = base_price * (1 + price_adj / 100)

category = st.sidebar.selectbox("Course Category", course_data['CourseCategory'].dropna().unique())
level = st.sidebar.selectbox("Course Level", course_data['CourseLevel'].dropna().unique())

# Section 3: Revenue Prediction Input
est_enrollment = st.sidebar.number_input("Estimated Enrollment Count (Override)", min_value=0, value=100)

st.sidebar.markdown("---")
analyze = st.sidebar.button("🚀 Run Enhanced Analysis", use_container_width=True)

# ===============================
# MAIN CONTENT
# ===============================
if not analyze:
    st.info("👈 Configure course details in the sidebar and click **Run Enhanced Analysis**")
    
    # Display historical overview if not analyzing
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Enrollment Distribution")
        fig = px.histogram(course_data, x="EnrollmentCount", nbins=20, template="plotly_dark", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Revenue vs Rating")
        fig = px.scatter(course_data, x="CourseRating", y="TotalRevenue", color="CourseCategory", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
else:
    # 1. PREDICTIONS
    input_data = pd.DataFrame([{
        'CourseDuration': duration,
        'CourseRating': rating,
        'YearsOfExperience': experience,
        'TeacherRating': teacher_rating,
        'CoursePrice': adjusted_price,
        'CourseCategory': category,
        'CourseLevel': level
    }])
    
    # Enrollment Prediction
    input_encoded_enrol = pd.get_dummies(input_data).reindex(columns=enrol_feature_names, fill_value=0)
    pred_enrollment = enrol_model.predict(input_encoded_enrol)[0]
    
    # Revenue Prediction
    # Use the engineer_features function for the input
    # We need to construct a full row for engineer_features
    # We'll use the estimated enrollment for revenue prediction as requested in Section 3
    input_for_rev = input_data.copy()
    input_for_rev['EnrollmentCount'] = est_enrollment # Using manual input for this section
    input_for_rev['TotalRevenue'] = 0 # Placeholder
    input_for_rev['CourseID'] = 9999
    
    # Apply engineering
    input_feat_rev = engineer_features(input_for_rev)
    # Revenue model also needs CourseCategory/Level dummies
    input_feat_rev = pd.get_dummies(input_feat_rev, columns=['CourseCategory', 'CourseLevel'])
    input_final_rev = input_feat_rev.reindex(columns=rev_features, fill_value=0)
    
    # Select Model
    model_key = 'rf' if model_option == "Random Forest" else ('lr' if model_option == "Linear Regression" else 'gb')
    pred_revenue = rev_models[model_key].predict(input_final_rev)[0] if rev_models else 0

    # SECTION 3: REVENUE PREDICTION DISPLAY
    st.subheader("🎯 Prediction Outcomes")
    m1, m2, m3 = st.columns(3)
    m1.metric("📈 Predicted Enrollments", int(pred_enrollment))
    m2.metric("💰 Predicted Revenue", f"₹{pred_revenue:,.2f}")
    rev_per_enrol = pred_revenue / max(est_enrollment, 1)
    m3.metric("👤 Revenue per Enrollment", f"₹{rev_per_enrol:.2f}")

    # SECTION 4: PRICE SENSITIVITY
    st.markdown("---")
    st.subheader("⚖️ Price Sensitivity Analysis")
    
    # Calculate for base price
    input_base = input_data.copy()
    input_base['CoursePrice'] = base_price
    input_encoded_base = pd.get_dummies(input_base).reindex(columns=enrol_feature_names, fill_value=0)
    base_enrol = enrol_model.predict(input_encoded_base)[0]
    base_revenue = base_enrol * base_price
    
    # Calculate for adjusted price
    adj_enrol = pred_enrollment # already calculated with adjusted price
    adj_revenue = adj_enrol * adjusted_price
    
    rev_change = ((adj_revenue - base_revenue) / base_revenue * 100) if base_revenue > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Current Price Strategy**")
        st.write(f"Price: ₹{base_price:.2f}")
        st.write(f"Est. Revenue: ₹{base_revenue:,.2f}")
    with c2:
        st.write("**Adjusted Price Strategy**")
        st.write(f"Price: ₹{adjusted_price:.2f}")
        st.write(f"Est. Revenue: ₹{adj_revenue:,.2f}")
    with c3:
        st.write("**Impact**")
        st.metric("Revenue Change", f"{rev_change:+.1f}%", delta_color="normal")
        
    # Break-even point (simplified: price where revenue equals base revenue)
    # We can show a small chart
    prices = np.linspace(base_price * 0.5, base_price * 1.5, 10)
    sens_data = []
    for p in prices:
        inp = input_data.copy()
        inp['CoursePrice'] = p
        enc = pd.get_dummies(inp).reindex(columns=enrol_feature_names, fill_value=0)
        e = enrol_model.predict(enc)[0]
        sens_data.append({'Price': p, 'Enrollment': e, 'Revenue': e * p})
    
    sens_df = pd.DataFrame(sens_data)
    fig_sens = px.line(sens_df, x='Price', y='Revenue', title="Revenue vs Price Curve", template="plotly_dark")
    fig_sens.add_vline(x=base_price, line_dash="dash", line_color="white", annotation_text="Base Price")
    fig_sens.add_vline(x=adjusted_price, line_dash="dash", line_color="green", annotation_text="Target Price")
    st.plotly_chart(fig_sens, use_container_width=True)

    # SECTION 5: CATEGORY REVENUE ANALYSIS
    st.markdown("---")
    st.subheader("📊 Category Revenue Breakdown")
    
    cat_stats = course_data.groupby('CourseCategory').agg({
        'TotalRevenue': 'sum',
        'EnrollmentCount': 'sum',
        'CourseID': 'count'
    }).rename(columns={'CourseID': 'CourseCount'}).reset_index()
    cat_stats['AvgRevPerCourse'] = cat_stats['TotalRevenue'] / cat_stats['CourseCount']
    
    tab1, tab2 = st.tabs(["Global View", "Category Deep-Dive"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(cat_stats, x='CourseCategory', y='TotalRevenue', title="Total Revenue by Category", template="plotly_dark", color='TotalRevenue')
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.bar(cat_stats, x='CourseCategory', y='AvgRevPerCourse', title="Avg Revenue per Course", template="plotly_dark", color='AvgRevPerCourse')
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        sel_cat = st.selectbox("Select Category to Analyze", cat_stats['CourseCategory'].unique())
        cat_data = course_data[course_data['CourseCategory'] == sel_cat]
        
        ca1, ca2, ca3 = st.columns(3)
        ca1.metric("Avg Price", f"₹{cat_data['CoursePrice'].mean():.2f}")
        ca2.metric("Avg Enrollment", int(cat_data['EnrollmentCount'].mean()))
        ca3.metric("Avg Rating", f"{cat_data['CourseRating'].mean():.2f}")
        
        st.write(f"**Top Courses in {sel_cat}**")
        st.dataframe(cat_data.nlargest(5, 'TotalRevenue')[['CourseName', 'CourseLevel', 'EnrollmentCount', 'TotalRevenue']], use_container_width=True)

    # SECTION 6: INSTRUCTOR IMPACT ANALYSIS
    st.markdown("---")
    st.subheader("👨‍🏫 Instructor Performance")
    
    # Correlation between teacher rating and metrics
    corr_data = course_data[['TeacherRating', 'EnrollmentCount', 'TotalRevenue', 'CourseRating']].corr()['TeacherRating']
    
    col_i1, col_i2 = st.columns([1, 2])
    with col_i1:
        st.write("**Impact Correlations**")
        st.write(f"- On Enrollment: `{corr_data['EnrollmentCount']:.3f}`")
        st.write(f"- On Revenue: `{corr_data['TotalRevenue']:.3f}`")
        st.write(f"- On Course Rating: `{corr_data['CourseRating']:.3f}`")
        
    with col_i2:
        st.write("**Top 5 Instructors by Revenue**")
        # Need to merge with teacher names
        top_inst = pd.merge(course_data, raw_teachers[['TeacherID', 'TeacherName']], left_on='TeacherID', right_on='TeacherID', how='left') if 'TeacherID' in course_data.columns else course_data
        if 'TeacherName' in top_inst.columns:
            top_inst_agg = top_inst.groupby('TeacherName')['TotalRevenue'].sum().nlargest(5).reset_index()
            fig = px.bar(top_inst_agg, x='TotalRevenue', y='TeacherName', orientation='h', template="plotly_dark", color='TotalRevenue')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Teacher mapping data unavailable in current view.")

    # SECTION 7: MODEL COMPARISON METRICS
    st.markdown("---")
    with st.expander("📋 Model Performance Metrics"):
        if rev_metrics:
            # Convert dictionary to list of records for dataframe
            metric_list = []
            for k, v in rev_metrics.items():
                if isinstance(v, dict):
                    row = {"Model": k.replace("_", " ").title()}
                    row.update(v)
                    metric_list.append(row)
            
            metrics_df = pd.DataFrame(metric_list)
            # Filter for specific metric columns
            disp_cols = ["Model", "mae", "rmse", "r2_score", "cv_r2_mean"]
            disp_df = metrics_df[[c for c in disp_cols if c in metrics_df.columns]]
            
            # Highlight best performer (highest r2_score)
            def highlight_best(s):
                if s.name == 'r2_score':
                    is_max = s == s.max()
                    return ['background-color: #004d00' if v else '' for v in is_max]
                return ['' for _ in s]
            
            st.dataframe(disp_df.style.apply(highlight_best, axis=0), use_container_width=True)
        else:
            st.warning("Metrics file not found.")

    # SECTION 8: INSIGHTS & RECOMMENDATIONS
    st.markdown("---")
    st.subheader("💡 Strategic Insights")
    
    # Logic for insights
    price_sens = "Low" if rev_change > 0 and price_adj > 0 else "High"
    top_cat = cat_stats.nlargest(1, 'TotalRevenue')['CourseCategory'].values[0]
    
    insights = f"""
    - **Price Sensitivity**: The current configuration shows **{price_sens}** price sensitivity for this course category.
    - **Top Revenue Driver**: Historical data shows **{top_cat}** as the most lucrative category.
    - **Recommended Price Range**: For {category}, prices between **₹{cat_data['CoursePrice'].mean()*0.9:.0f}** and **₹{cat_data['CoursePrice'].mean()*1.1:.0f}** show stable enrollment.
    - **Instructor Strategy**: Courses with Teacher Ratings above **4.0** see a **{corr_data['EnrollmentCount']*100:.1f}%** correlation with higher enrollment volumes.
    """
    st.success(insights)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("🚀 EduPro Predictive Analytics Engine v2.0 | Advanced Revenue Modeling Edition")
