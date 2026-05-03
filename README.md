# Real-estate-price-prediction
Real Estate Price Prediction using Linear Regression, Random Forest &amp; Gradient Boosting
# Real Estate Price Prediction — Regression Modelling

## 📌 Overview
Built and compared 4 machine learning models to predict house prices 
on a dataset of 2,930 records with 17 features including location, 
quality, area, and age.

## 🛠️ Tools & Libraries
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

## 🤖 Models Compared
| Model | R² Score | MAE |
|-------|----------|-----|
| Linear Regression | 0.663 | $60,053 |
| Ridge Regression | 0.663 | $60,030 |
| Random Forest | 0.947 | $22,516 |
| Gradient Boosting | 0.971 | $16,902 ✅ Best |

## 📊 Key Findings
- Gradient Boosting achieved R² = 0.971 (97.1% accuracy)
- Top price drivers: Neighborhood, Living Area, Quality × Area
- Each quality point increase → ~$12,409 price increase
- Ensemble models outperform linear by 30.8 percentage points

## 📁 Files
| File | Description |
|------|-------------|
| real_estate_analysis.py | Main analysis + ML script |
| housing_data.csv | Dataset (2,930 records) |
| figure1_eda_dashboard.png | EDA dashboard |
| figure2_model_performance.png | Model comparison |
| figure3_business_insights.png | Business insights |

## ▶️ How to Run
pip install pandas numpy matplotlib seaborn scikit-learn
python real_estate_analysis.py
