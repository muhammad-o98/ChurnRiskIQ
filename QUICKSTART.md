# 🚀 Quick Start Guide - Telecom Churn Prediction App

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation & Launch (3 Steps)

### Step 1: Install Dependencies
```bash
cd /Users/ob/telecom-churn
pip install -r requirements.txt
```

### Step 2: Launch the App
```bash
streamlit run app.py
```

### Step 3: Open Your Browser
The app will automatically open at: **http://localhost:8501**

---

## 📱 How to Use the App

### 1️⃣ **Data Upload** (Page 1)
- Click "📤 Data Upload" in the sidebar
- Upload your CSV file OR use the sample dataset
- Review preprocessing results and visualizations

### 2️⃣ **Model Training** (Page 2)
- Click "🤖 Model Training" in the sidebar
- Select one or more models to train
- View comparison metrics and ROC curves

### 3️⃣ **SHAP Analysis** (Page 3)
- Click "🔍 SHAP Analysis" in the sidebar
- Compute SHAP values (may take ~30-60 seconds)
- Explore feature importance and top churn drivers

### 4️⃣ **Risk Segmentation** (Page 4)
- Click "⚠️ Risk Segmentation" in the sidebar
- Generate customer risk segments
- Review retention strategies
- Export high-risk customer lists

### 5️⃣ **Dashboard** (Page 5)
- Click "📊 Dashboard" in the sidebar
- View executive summary with KPIs
- Download reports

---

## 💡 Tips

- **Session State**: The app remembers your data and models across pages
- **Sample Data**: Use the built-in sample data if you don't have a dataset
- **Export**: Download CSV files and reports from each page
- **Best Model**: The app automatically selects the best model based on PR AUC

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### SHAP Analysis Taking Too Long
- This is normal for KernelExplainer (~30-60 seconds)
- The app will show progress bars

---

## 📊 Data Format

Your CSV should have these columns (or similar):
- **Churn**: Target variable (Yes/No)
- **tenure**: Customer tenure in months
- **MonthlyCharges**: Monthly charges
- **Contract**: Contract type (Month-to-month, One year, Two year)
- **InternetService**: Internet service type
- Other demographic and service features

Sample data is provided at: `data/churndata.csv`

---

## 🎯 Next Steps

1. **Train Models**: Start with Random Forest and XGBoost for quick results
2. **Analyze SHAP**: Understand which features drive churn
3. **Segment Customers**: Identify high-risk customers for retention campaigns
4. **Export Reports**: Download customer lists and share insights

---

## 📞 Need Help?

- Check the main README.md for detailed documentation
- Review the Jupyter notebook at: `notebooks/telecom_churn.ipynb`
- Inspect the code in `utils/` for data processing and model logic

Happy Analyzing! 🎉
