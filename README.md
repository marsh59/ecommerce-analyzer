# 📊 EcomAnalyzer — E-Commerce Performance Dashboard

A full-stack data analytics web application built with **Flask**, **Pandas**, and **Plotly**.  
Upload **UCI Online Retail Dataset** and instantly explore interactive dashboards for sales, products, customers, forecasting, and RFM segmentation.



## 📸 Features

| Dashboard | What it shows |
|---|---|
| 💰 Sales | Monthly revenue trend, sales by country |
| 📦 Products | Top products by revenue, revenue by category |
| 👤 Customers | Top customers, repeat orders, customers by country |
| 🔮 Forecast | 6-month sales prediction with confidence range |
| 🎯 RFM Analysis | Customer segmentation — Champions, Loyal, At Risk, Lost |

---

## 🛠️ Built With

- **Python** — core language
- **Flask** — web framework
- **Pandas** — data cleaning and analysis
- **Plotly** — interactive charts
- **Statsmodels** — Holt-Winters sales forecasting
- **PyMySQL** — MySQL database connector


## working of this project

## 📋 Dataset

This app uses the **UCI Online Retail Dataset** — a UK-based online retail dataset covering transactions from December 2010 to December 2011.

- **Period:** December 2010 to December 2011
- **Transactions:** 500,000+
- **Customers:** ~4,300 unique customers
- **Countries:** 38

### ⬇️ How to get the dataset

The dataset file `online_retail.csv` is included in this repository or you  can download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

**Steps:**
1. Go to this repository on GitHub
2. Click on the file `online_retail.csv`
3. Click the **Download raw file** button (top right, looks like a download arrow)
4. Save it anywhere on your computer

### 📤 How to use it in the app

1. Open the app in your browser
2. On the home page, click **Upload CSV**
3. Select the `online_retail.csv` file you just downloaded
4. Click **Upload & Analyze**
5. All dashboards will now be ready to explore