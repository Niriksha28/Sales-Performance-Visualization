# Sales Performance Visualization 📊

A Python project for analyzing sales team performance using pandas, train_test_split, and StandardScaler.

## Features
- Generates realistic sales data for 100 salespeople
- Creates 9 comprehensive visualizations (revenue distribution, regional analysis, correlation heatmaps)
- Machine learning model to predict sales revenue with feature importance
- Business insights on top performers and regional trends

## Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
```python
python sales_performance_visualization.py
```

## Output
- **Visualizations**: 9 charts analyzing sales metrics, correlations, and performance
- **ML Model**: Revenue prediction with R² score ~0.85 and feature importance rankings
- **Insights**: Top performer analysis, regional comparisons, success factor identification

## Data Columns
`name`, `experience_years`, `age`, `region`, `calls_per_day`, `meetings_per_week`, `conversion_rate`, `sales_revenue`, `customer_satisfaction`

## Key Insights Generated
- What makes top 10% performers successful
- Regional performance differences
- Correlation between experience, conversion rates, and revenue
- Feature importance for sales success prediction

Ready to analyze sales performance with data science! 🚀
