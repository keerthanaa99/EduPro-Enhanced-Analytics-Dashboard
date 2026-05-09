# Data Dictionary - EduPro Course Demand Prediction

## Original Features

| Feature | Type | Description | Range | Source |
| :--- | :--- | :--- | :--- | :--- |
| **CourseID** | String | Unique course identifier | CR00001 - CR00060 | Courses table |
| **CourseName** | String | Course title | Variable | Courses table |
| **CourseCategory** | Categorical | Subject area | Programming, Data Science, Web Dev, etc. | Courses table |
| **CourseType** | String | Course format | Online, Hybrid, Instructor-led | Courses table |
| **CourseLevel** | Categorical | Difficulty level | Beginner, Intermediate, Advanced | Courses table |
| **CoursePrice** | Numeric (USD) | Listed price | $25 - $249 | Courses table |
| **CourseDuration** | Numeric (hours) | Total course hours | 5 - 60 hours | Courses table |
| **CourseRating** | Numeric | Student rating (avg) | 1.0 - 5.0 stars | Courses table |
| **TeacherID** | String | Unique instructor ID | TC00001 - TC00060 | Teachers table |
| **TeacherName** | String | Instructor name | Variable | Teachers table |
| **YearsOfExperience** | Numeric | Years teaching | 1 - 30 years | Teachers table |
| **TeacherRating** | Numeric | Instructor rating (avg) | 1.0 - 5.0 stars | Teachers table |
| **Expertise** | String | Primary teaching specialty | Variable | Teachers table |
| **TransactionID** | String | Unique enrollment ID | TT00001 - TT10000 | Transactions table |
| **TransactionDate** | Date | Enrollment date | 2023-01-01 - 2024-12-31 | Transactions table |
| **Amount** | Numeric (USD) | Revenue per enrollment | $0 - $500 | Transactions table |
| **UserID** | String | Student ID | U00001 - U03000 | Users table |
| **EnrollmentCount** | Numeric | Total enrollments per course | 50 - 245 | Aggregated from Transactions |
| **TotalRevenue** | Numeric (USD) | Total revenue per course | $1,250 - $45,000 | Aggregated from Transactions |

## Engineered Features

| Feature | Type | Description | Values | Created By |
| :--- | :--- | :--- | :--- | :--- |
| **PriceBand** | Categorical | Price range | Low (<$50), Medium ($50-$150), High (>$150) | `feature_engineering.py` |
| **DurationBucket** | Categorical | Duration range | Short (<10h), Medium (10-30h), Long (>30h) | `feature_engineering.py` |
| **RatingTier** | Categorical | Course quality tier | Poor (<2.5), Average (2.5-3.5), Excellent (>3.5) | `feature_engineering.py` |
| **ExperienceBucket** | Categorical | Teacher experience level | Junior (<5yr), Mid (5-15yr), Senior (>15yr) | `feature_engineering.py` |
| **RevenuePerEnrollment** | Numeric (USD) | Revenue efficiency | Calculated: TotalRevenue / EnrollmentCount | `feature_engineering.py` |
| **TeacherRatingCategory** | Categorical | Instructor quality | Low (<2.5), Good (2.5-4.0), Excellent (>4.0) | `feature_engineering.py` |
| **PopularityScore** | Numeric | Normalized enrollments | 0 - 100 (normalized within category) | `feature_engineering.py` |

## Target Variables

| Variable | Type | Description | Range | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **EnrollmentCount** | Numeric | Total students per course | 50 - 245 | Primary prediction target |
| **TotalRevenue** | Numeric (USD) | Total revenue per course | $1,250 - $45,000 | Secondary prediction target |
| **CategoryRevenue** | Numeric (USD) | Aggregate revenue by category | $15,000 - $180,000 | Portfolio analysis |

## Data Quality Notes
- **Missing Values:** Handled via median imputation for ratings and forward-fill for dates.
- **Outliers:** Retained for realism; high-revenue courses represent "Blockbuster" content.
- **Imbalanced Data:** Categories like "Data Science" and "Programming" have naturally higher density.
- **Aggregation:** All models operate at the **Course Level** rather than the Transaction Level for strategic forecasting.

## Statistical Summary (Course Level)
| Statistic | EnrollmentCount | TotalRevenue | CoursePrice | CourseDuration |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | 166.7 | $20,750 | $124.50 | 28.4 |
| **Std Dev** | 42.3 | $8,400 | $45.20 | 12.1 |
| **Min** | 50 | $1,250 | $25.00 | 5.0 |
| **Max** | 245 | $45,000 | $249.00 | 60.0 |
| **Median** | 162 | $19,800 | $115.00 | 26.5 |

## Encoding Scheme
### Categorical Variables (One-Hot Encoded)
The following features are expanded into binary (0/1) columns during preprocessing:
- `CourseCategory`, `CourseLevel`
- `PriceBand`, `DurationBucket`, `RatingTier`
- `ExperienceBucket`, `TeacherRatingCategory`

### Numerical Scaling
- **Min-Max Scaling:** Applied to the `PopularityScore` within each category to ensure a 0-100 range.
- **Raw Scaling:** Other numerical features are kept in their original units (hours, USD, stars) to maintain model interpretability in the dashboard.

## Usage Examples
### Predicting New Course Viability
- **Input:** `CoursePrice=99`, `CourseDuration=20`, `CourseCategory='Data Science'`
- **Model Output:** Predicted `EnrollmentCount` and `TotalRevenue`.

### Sensitivity Analysis
- Adjust `CoursePrice` in the dashboard to see how the `PriceBand` and total revenue shift.

## Related Files
- **Dataset:** `EduPro Online Platform.xlsx`
- **Logic:** `feature_engineering.py`
- **Analytics:** `revenue_models.py`, `evaluate_models.py`
- **Interface:** `app_enhanced.py`
