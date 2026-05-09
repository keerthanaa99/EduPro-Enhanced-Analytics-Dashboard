import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from feature_engineering import engineer_features

def load_and_preprocess_data(file_path):
    # 1. LOAD DATA
    print(f"Loading data from {file_path}...")
    xl = pd.read_excel(file_path, sheet_name=None)
    
    courses_df = xl['Courses']
    teachers_df = xl['Teachers']
    transactions_df = xl['Transactions']
    
    # 2. AGGREGATE AT COURSE LEVEL
    # First, get enrollment and revenue from transactions
    course_stats = transactions_df.groupby('CourseID').agg({
        'TransactionID': 'count',
        'Amount': 'sum',
        'TeacherID': 'first' # Assuming one primary teacher per course for modeling
    }).rename(columns={'TransactionID': 'EnrollmentCount', 'Amount': 'TotalRevenue'})
    
    # Merge with Courses info
    course_data = pd.merge(courses_df, course_stats, on='CourseID', how='left')
    
    # Merge with Teachers info
    course_data = pd.merge(course_data, teachers_df, on='TeacherID', how='left')
    
    # Handle missing values for courses with no transactions
    course_data['EnrollmentCount'] = course_data['EnrollmentCount'].fillna(0)
    course_data['TotalRevenue'] = course_data['TotalRevenue'].fillna(0)
    
    return course_data

def train_revenue_models(df):
    # Apply feature engineering
    print("Applying feature engineering...")
    df_feat = engineer_features(df)
    
    # Target variables
    target = 'TotalRevenue'
    
    # Define features
    # Based on task: [CourseDuration, CourseRating, YearsOfExperience, TeacherRating, 
    # CourseCategory, CourseLevel, CoursePrice, EnrollmentCount]
    # Plus the engineered ones. We need to ensure CourseCategory and CourseLevel are encoded.
    
    # Identify categorical columns to encode if not already handled
    categorical_cols = ['CourseCategory', 'CourseLevel']
    df_feat = pd.get_dummies(df_feat, columns=categorical_cols, prefix=categorical_cols)
    
    # Drop non-feature columns
    # We should keep only numeric columns for modeling (after one-hot encoding)
    X = df_feat.select_dtypes(include=[np.number, 'bool'])
    
    # Target is TotalRevenue
    if target in X.columns:
        X = X.drop(columns=[target])
    
    # Also drop CourseID if it's still there (it's numeric but not a feature)
    if 'CourseID' in X.columns:
        X = X.drop(columns=['CourseID'])
    
    y = df_feat[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    results = []
    trained_models = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        })
        trained_models[name] = model
        
        # Save models
        short_name = name.lower().replace(" ", "_")
        joblib.dump(model, f"{short_name}_revenue_model.pkl")
        
    # Save metrics for app
    results_df = pd.DataFrame(results)
    results_df.to_json('metrics.json', orient='records')
    
    # Save feature names for later use
    joblib.dump(X.columns.tolist(), 'revenue_model_features.pkl')
    
    return trained_models, results_df, X_train, y_train

def analyze_category_performance(df):
    # 5. CATEGORY-LEVEL AGGREGATION
    cat_agg = df.groupby('CourseCategory').agg({
        'TotalRevenue': 'sum',
        'EnrollmentCount': 'sum',
        'CourseRating': 'mean',
        'CoursePrice': 'mean'
    }).reset_index()
    
    cat_agg['RevenuePerEnrollment'] = cat_agg['TotalRevenue'] / cat_agg['EnrollmentCount'].replace(0, np.nan)
    cat_agg = cat_agg.fillna(0)
    
    return cat_agg

def main():
    file_path = 'EduPro Online Platform.xlsx'
    course_data = load_and_preprocess_data(file_path)
    
    # Train and evaluate
    trained_models, results_df, X_train, y_train = train_revenue_models(course_data)
    
    # 4. Evaluation Table
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    # 6. FEATURE IMPORTANCE (Random Forest)
    rf_model = trained_models['Random Forest']
    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    feat_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Features Influencing Revenue:")
    print(feat_importance_df.head(5).to_string(index=False))
    
    # 5. Category Aggregation
    cat_performance = analyze_category_performance(course_data)
    print("\nCategory Performance Summary:")
    print(cat_performance.to_string(index=False))
    
    # 8. Optimization Opportunities
    top_feature = feat_importance_df.iloc[0]['Feature']
    print(f"\n--- REVENUE OPTIMIZATION OPPORTUNITIES ---")
    print(f"1. Key Driver: {top_feature} has the highest impact on revenue.")
    
    # Identify high revenue but low enrollment categories (Premium potential)
    premium_cats = cat_performance[cat_performance['RevenuePerEnrollment'] > cat_performance['RevenuePerEnrollment'].mean()]
    print(f"2. Premium Categories (High Rev/Enrollment): {', '.join(premium_cats['CourseCategory'].tolist())}")
    
    # Identify high enrollment but low revenue categories (Upsell potential)
    mass_cats = cat_performance[(cat_performance['EnrollmentCount'] > cat_performance['EnrollmentCount'].mean()) & 
                                (cat_performance['TotalRevenue'] < cat_performance['TotalRevenue'].mean())]
    if not mass_cats.empty:
        print(f"3. Upsell Potential: Categories like {', '.join(mass_cats['CourseCategory'].tolist())} have high volume but low revenue. Consider adjusting pricing.")

if __name__ == "__main__":
    main()
