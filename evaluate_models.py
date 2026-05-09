import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                             mean_absolute_percentage_error, median_absolute_error)
from scipy import stats
from feature_engineering import engineer_features

def load_and_prepare_data(file_path):
    # Load raw data
    xl = pd.read_excel(file_path, sheet_name=None)
    courses = xl['Courses']
    teachers = xl['Teachers']
    transactions = xl['Transactions']
    
    # Merge
    data = pd.merge(transactions, courses, on="CourseID", how="left")
    data = pd.merge(data, teachers, on="TeacherID", how="left")
    
    # Course-level aggregation
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
    
    course_data = pd.merge(course_data, courses[['CourseID', 'CourseCategory', 'CourseLevel']], on='CourseID', how='left')
    
    # Feature Engineering
    df_feat = engineer_features(course_data)
    
    # One-hot encode original categories
    df_feat = pd.get_dummies(df_feat, columns=['CourseCategory', 'CourseLevel'], prefix=['Cat', 'Lvl'])
    
    # Select numeric features
    X = df_feat.select_dtypes(include=[np.number, 'bool'])
    target = 'EnrollmentCount'
    
    if target in X.columns:
        X = X.drop(columns=[target, 'CourseID', 'TotalRevenue'], errors='ignore')
    
    y = df_feat[target]
    
    return X, y

def evaluate_models():
    file_path = 'EduPro Online Platform.xlsx'
    X, y = load_and_prepare_data(file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42)
    }
    
    all_metrics = {}
    best_r2 = -float('inf')
    best_model_name = ""
    
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10} | {'CV R² (mean±std)':<20}")
    print("-" * 80)
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Basic Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        med_ae = median_absolute_error(y_test, y_pred)
        
        # Cross-Validation (5-fold)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store
        all_metrics[name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "mape": round(mape, 4),
            "median_absolute_error": round(med_ae, 4),
            "cv_r2_mean": round(cv_mean, 4),
            "cv_r2_std": round(cv_std, 4)
        }
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            
        print(f"{name.replace('_', ' ').title():<20} | {mae:<10.2f} | {rmse:<10.2f} | {r2:<10.3f} | {cv_mean:.3f} ±{cv_std:.3f}")
        
        # Save model pkl
        joblib.dump(model, f"{name}_enrollment_model.pkl")

    # Add metadata to metrics
    all_metrics["best_model"] = best_model_name
    all_metrics["test_size"] = 0.2
    all_metrics["training_samples"] = len(X_train)
    all_metrics["test_samples"] = len(X_test)
    
    # Save metrics.json
    with open('metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    # PREDICTION EXAMPLES (Section 4)
    print("\n--- PREDICTION EXAMPLES (Random 5 Test Samples) ---")
    print(f"{'Actual':<10} | {'Predicted':<10} | {'Error (%)':<10}")
    print("-" * 40)
    
    test_indices = np.random.choice(len(y_test), 5, replace=False)
    rf_preds = models['random_forest'].predict(X_test)
    y_test_array = y_test.values
    
    for idx in test_indices:
        actual = y_test_array[idx]
        predicted = rf_preds[idx]
        error_pct = (abs(actual - predicted) / actual * 100) if actual > 0 else 0
        print(f"{actual:<10.0f} | {predicted:<10.2f} | {error_pct:.1f}%")
        
    # FEATURE IMPORTANCE (Section 7)
    rf = models['random_forest']
    feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_10_feats = feat_importances.sort_values(ascending=False).head(10)
    
    print("\n--- TOP 10 FEATURES (Random Forest) ---")
    print(top_10_feats)
    
    with open('feature_importance.json', 'w') as f:
        json.dump(top_10_feats.to_dict(), f, indent=4)
        
    # RESIDUAL ANALYSIS (Section 8)
    residuals = y_test_array - rf_preds
    shapiro_test = stats.shapiro(residuals)
    
    print("\n--- RESIDUAL ANALYSIS ---")
    print(f"Mean Residual: {np.mean(residuals):.4f}")
    print(f"Residual Std Dev: {np.std(residuals):.4f}")
    print(f"Shapiro-Wilk Test (p-value): {shapiro_test.pvalue:.4f}")
    
    if shapiro_test.pvalue > 0.05:
        print("Residuals appear to be normally distributed.")
    else:
        print("Residuals do not follow a normal distribution (check for outliers or non-linear patterns).")

if __name__ == "__main__":
    evaluate_models()
