import pandas as pd
import numpy as np

def engineer_features(course_data):
    """
    Engineers features for the EduPro course demand prediction project.
    
    Parameters:
    course_data (pd.DataFrame): Raw dataframe with columns:
        CourseID, CourseDuration, CourseRating, YearsOfExperience, TeacherRating, 
        CourseCategory, CourseLevel, EnrollmentCount, CoursePrice, TotalRevenue
        
    Returns:
    pd.DataFrame: Augmented dataframe with 7 new features and one-hot encoding.
    """
    # Work on a copy to avoid modifying the original dataframe
    df = course_data.copy()
    
    # Ensure numerical columns are numeric, handling potential strings/NaNs
    num_cols = ['CourseDuration', 'CourseRating', 'YearsOfExperience', 'TeacherRating', 
                'EnrollmentCount', 'CoursePrice', 'TotalRevenue']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. Price Bands (from CoursePrice)
    # Low (< 50), Medium (50-150), High (> 150)
    df['PriceBand'] = pd.cut(
        df['CoursePrice'], 
        bins=[-np.inf, 49.99, 150, np.inf], 
        labels=['Low', 'Medium', 'High']
    )

    # 2. Duration Buckets (from CourseDuration)
    # Short (< 10), Medium (10-30), Long (> 30)
    df['DurationBucket'] = pd.cut(
        df['CourseDuration'], 
        bins=[-np.inf, 9.99, 30, np.inf], 
        labels=['Short', 'Medium', 'Long']
    )

    # 3. Rating Tiers (from CourseRating)
    # Poor (< 2.5), Average (2.5-3.5), Excellent (> 3.5)
    df['RatingTier'] = pd.cut(
        df['CourseRating'], 
        bins=[-np.inf, 2.49, 3.5, np.inf], 
        labels=['Poor', 'Average', 'Excellent']
    )

    # 4. Experience Buckets (from YearsOfExperience)
    # Junior (< 5), Mid-level (5-15), Senior (> 15)
    df['ExperienceBucket'] = pd.cut(
        df['YearsOfExperience'], 
        bins=[-np.inf, 4.99, 15, np.inf], 
        labels=['Junior', 'Mid-level', 'Senior']
    )

    # 5. Revenue Per Enrollment
    # Formula: TotalRevenue / EnrollmentCount (Handle division by zero)
    # If EnrollmentCount is 0 or NaN, result is 0
    df['RevenuePerEnrollment'] = df.apply(
        lambda x: x['TotalRevenue'] / x['EnrollmentCount'] if x['EnrollmentCount'] > 0 else 0, 
        axis=1
    )

    # 6. Teacher Rating Category
    # Low (< 2.5), Good (2.5-4.0), Excellent (> 4.0)
    df['TeacherRatingCategory'] = pd.cut(
        df['TeacherRating'], 
        bins=[-np.inf, 2.49, 4.0, np.inf], 
        labels=['Low', 'Good', 'Excellent']
    )

    # 7. Popularity Score (engineered)
    # Formula: (EnrollmentCount - category_min) / (category_max - category_min) * 100
    if 'CourseCategory' in df.columns:
        def calculate_popularity(group):
            c_min = group.min()
            c_max = group.max()
            if c_max == c_min:
                return group * 0 # Or 100? Setting 0 for stability
            return (group - c_min) / (c_max - c_min) * 100
        
        df['PopularityScore'] = df.groupby('CourseCategory')['EnrollmentCount'].transform(calculate_popularity)
    else:
        df['PopularityScore'] = 0

    # Handle NaNs in engineered features before encoding
    engineered_cats = ['PriceBand', 'DurationBucket', 'RatingTier', 'ExperienceBucket', 'TeacherRatingCategory']
    for col in engineered_cats:
        # Fill NaNs with a placeholder or mode. Here we use 'Unknown' or similar if needed, 
        # but pd.get_dummies can handle it.
        df[col] = df[col].astype(object).fillna('Missing')

    # Include one-hot encoding for the categorical engineered features
    df = pd.get_dummies(df, columns=engineered_cats, prefix=engineered_cats)

    return df

if __name__ == "__main__":
    # Example usage
    data = {
        'CourseID': [1, 2, 3, 4, 5],
        'CourseDuration': [5, 15, 35, 10, 20],
        'CourseRating': [2.0, 3.0, 4.5, 3.2, 1.5],
        'YearsOfExperience': [3, 10, 20, 8, 2],
        'TeacherRating': [2.0, 3.5, 4.8, 3.0, 1.2],
        'CourseCategory': ['Tech', 'Tech', 'Design', 'Design', 'Tech'],
        'CourseLevel': ['Beginner', 'Intermediate', 'Advanced', 'Beginner', 'Beginner'],
        'EnrollmentCount': [100, 200, 50, 60, 10],
        'CoursePrice': [30, 80, 200, 120, 40],
        'TotalRevenue': [3000, 16000, 10000, 7200, 400]
    }
    
    sample_df = pd.DataFrame(data)
    print("Original Data (First 5 rows):")
    print(sample_df)
    
    engineered_df = engineer_features(sample_df)
    
    print("\nEngineered Data (Columns):")
    print(engineered_df.columns.tolist())
    
    print("\nSample Rows with new features:")
    # Showing a few columns for brevity
    cols_to_show = ['CourseID', 'RevenuePerEnrollment', 'PopularityScore'] + \
                   [c for c in engineered_df.columns if 'PriceBand' in c]
    print(engineered_df[cols_to_show].head())
