import pandas as pd
import numpy as np

# ===============================
# STEP 1: LOAD DATA
# ===============================
file_path = "EduPro Online Platform.xlsx"

users = pd.read_excel(file_path, sheet_name="Users")
courses = pd.read_excel(file_path, sheet_name="Courses")
teachers = pd.read_excel(file_path, sheet_name="Teachers")
transactions = pd.read_excel(file_path, sheet_name="Transactions")


# ===============================
# STEP 2: MERGE DATA
# ===============================
data = pd.merge(transactions, courses, on="CourseID", how="left")
data = pd.merge(data, teachers, on="TeacherID", how="left")


# ===============================
# STEP 3: PREPROCESSING
# ===============================
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
data['Month'] = data['TransactionDate'].dt.month


# ===============================
# STEP 4: AGGREGATE (COURSE LEVEL)
# ===============================
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

# Merge course info
course_data = pd.merge(course_data, courses, on='CourseID', how='left')


# ===============================
# FIX COLUMN NAMES (IMPORTANT)
# ===============================
course_data.rename(columns={
    'CourseDuration_x': 'CourseDuration',
    'CourseRating_x': 'CourseRating'
}, inplace=True)


print(course_data.head())
print("Shape:", course_data.shape)


# ===============================
# STEP 5: FEATURE SELECTION
# ===============================
features = [
    'CourseDuration',
    'CourseRating',
    'YearsOfExperience',
    'TeacherRating',
    'CourseCategory',
    'CourseLevel'
]

target = 'EnrollmentCount'


# ===============================
# STEP 6: ENCODING
# ===============================
course_encoded = pd.get_dummies(course_data[features], drop_first=True)

X = course_encoded
y = course_data[target]


# ===============================
# STEP 7: TRAIN-TEST SPLIT
# ===============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/Test Shapes:", X_train.shape, X_test.shape)


# ===============================
# STEP 8: LINEAR REGRESSION
# ===============================
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)


# ===============================
# STEP 9: RANDOM FOREST
# ===============================
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


# ===============================
# STEP 10: EVALUATION
# ===============================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\nLinear Regression Results:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2 Score:", r2_score(y_test, y_pred_lr))

print("\nRandom Forest Results:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2 Score:", r2_score(y_test, y_pred_rf))


# ===============================
# STEP 11: FEATURE IMPORTANCE
# ===============================
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nFeature Importance:\n")
print(importance)