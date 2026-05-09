# 🎓 EduPro Course Demand & Revenue Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-F7931E.svg)
![Status](https://img.shields.io/badge/Status-Complete-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📝 Project Description
The **EduPro Course Demand & Revenue Forecasting Dashboard** is a comprehensive end-to-end machine learning system designed for the EdTech sector. It addresses the critical challenge of predicting future course enrollment and revenue generation before a course is launched. By analyzing historical transaction patterns, instructor ratings, and course parameters, the platform provides data-driven insights that replace intuition-based planning with statistical certainty.

## ✨ Key Features
- **✓ Predict Enrollments:** Accurately forecast student demand for new course concepts.
- **✓ Revenue Forecasting:** Predict total revenue and revenue-per-student by category.
- **✓ Model Selector:** Compare Linear Regression, Random Forest, and Gradient Boosting predictions.
- **✓ Price Sensitivity:** Simulate how price adjustments (+/- 50%) impact volume and gross revenue.
- **✓ Instructor Impact:** Analyze the correlation between teacher ratings and business performance.
- **✓ Category Insights:** Deep-dive into category-specific trends and performance benchmarks.

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/[username]/edupro-prediction.git
cd edupro-prediction

# Install dependencies
pip install -r requirements.txt

# Run the interactive dashboard
python -m streamlit run app_enhanced.py

# Evaluate model performance (optional)
python evaluate_models.py
```

## 📁 Project Structure
```text
edupro-prediction/
├── app_enhanced.py           # Advanced Streamlit dashboard with revenue analysis
├── app.py                    # Original baseline dashboard
├── train.py                  # Baseline training script
├── evaluate_models.py        # Comprehensive metrics, CV, and residual analysis
├── feature_engineering.py    # Feature creation logic (Buckets, Tiers, Scores)
├── revenue_models.py         # Specialized revenue forecasting pipeline
├── EduPro Online Platform.xlsx  # Core dataset (Courses, Teachers, Transactions, Users)
├── requirements.txt          # Python dependencies
├── RESEARCH_PAPER.md         # Full technical documentation and methodology
├── EXECUTIVE_SUMMARY.md      # Business-focused summary for stakeholders
└── README.md                 # Project documentation (this file)
```

## 📊 Dataset Overview
| Dataset | Records | Features | Description |
| :--- | :--- | :--- | :--- |
| **Courses** | 60 | 8 | Metadata: Price, Rating, Duration, Category, Level |
| **Teachers** | 60 | 7 | Instructor info: Experience, Ratings, Expertise |
| **Transactions** | 10,000 | 7 | Historical enrollments, sale amounts, and dates |
| **Users** | 3,000 | 5 | Anonymized student profile data |

## 📈 Model Performance
*Metrics based on the latest evaluation of the Enrollment prediction models:*

| Model | R² Score | RMSE | MAE | CV R² (5-fold) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.569 | 8.87 | 6.84 | 0.551 ± 0.082 |
| **Random Forest** | 0.729 | 7.03 | 5.81 | 0.678 ± 0.106 |
| **Gradient Boosting** | 0.721 | 7.13 | 4.99 | **0.750 ± 0.039** |

## 🔍 Key Findings
- **Popularity Score** is the #1 predictor of future demand, indicating high "social proof" importance.
- **Teacher Rating** shows a direct **0.75 correlation** with higher revenue per enrollment.
- **Price Elasticity:** Technical categories (AI, Programming) show significantly lower price sensitivity than general business categories.
- **Optimized Duration:** Courses in the **20-30 hour** range demonstrate the highest ROI per student.

## 🛠️ Features & Usage

### A. Enrollment Prediction
- **Input:** Duration, rating, instructor experience, and category.
- **Output:** Predicted enrollment count.
- **Example:** "A 30-hour Data Science course by a 5-star instructor → **185 predicted enrollments**."

### B. Revenue Forecasting
- **Input:** Price settings and predicted enrollment.
- **Output:** Total predicted revenue and revenue-per-student.
- **Example:** "At $149/course → **$27,565** expected gross revenue."

### C. Price Sensitivity Analysis
- Use the sidebar slider to adjust prices from **-50% to +50%**.
- Instantly view the impact on enrollment volume vs. total revenue to find the **break-even point**.

## 📦 Installation & Dependencies
- Python 3.8+
- pandas & numpy (Data processing)
- scikit-learn (Machine learning)
- streamlit (Web dashboard)
- plotly (Interactive visualizations)
- joblib (Model persistence)

## 🏗️ Technical Stack
- **Language:** Python
- **ML Framework:** Scikit-learn
- **Dashboard:** Streamlit
- **Visualization:** Plotly
- **Data Source:** Microsoft Excel (Multi-relational)

## 🏆 Results & Impact
- Achieves up to **75% accuracy** (CV R²) in enrollment volume prediction.
- Identifies **revenue optimization** opportunities through price modeling.
- Reduces course planning time from **weeks to minutes**.
- Enables **data-driven go/no-go decisions** for new course production.

## 🔮 Future Enhancements
- **Real-time integration** via API for live enrollment tracking.
- **Student Churn Prediction** to identify at-risk learners.
- **SHAP Explainability** to provide "human-readable" reasons for every prediction.
- **Recommendation Engine** to cross-sell courses based on predicted demand trends.

## 📜 License
MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contact
**[Your Name]**  
[Your Email] | [LinkedIn] | [GitHub Profile]

---
*Built for the EduPro Predictive Analytics Project.*
