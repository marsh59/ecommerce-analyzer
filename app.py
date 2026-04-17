from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import plotly.express as px

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

df = None

@app.route('/')
def home():
    return render_template('index.html')


# 📂 Upload CSV
@app.route('/upload', methods=['POST'])
def upload():
    global df

    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    df = pd.read_csv(path, encoding='latin1')

    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]

    df['CustomerID'] = df['CustomerID'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Total'] = df['Quantity'] * df['UnitPrice']

    return jsonify({"message": "✅ File uploaded successfully!"})


# 💰 SALES DASHBOARD
@app.route('/sales')
def sales():
    global df

    if df is None:
        return "❌ Upload file first!"

    total_sales = round(df['Total'].sum(), 2)
    total_orders = df['InvoiceNo'].nunique()

    # 📈 Trend
    trend = df.groupby('InvoiceDate')['Total'].sum().reset_index()
    fig1 = px.line(trend, x='InvoiceDate', y='Total',
                   title="📈 Sales Trend Over Time")

    trend_chart = fig1.to_html(full_html=False)

    # 🌍 Country
    country = df.groupby('Country')['Total'].sum().reset_index() \
                .sort_values(by='Total', ascending=False).head(10)

    fig2 = px.pie(country, names='Country', values='Total',
                  title="🌍 Sales by Country")

    country_chart = fig2.to_html(full_html=False)

    return render_template('sales.html',
                           total_sales=total_sales,
                           total_orders=total_orders,
                           trend_chart=trend_chart,
                           country_chart=country_chart)


# 📦 PRODUCTS DASHBOARD
@app.route('/products')
def products():
    global df

    if df is None:
        return "❌ Upload file first!"

    product = df.groupby('Description')['Total'].sum().reset_index() \
                .sort_values(by='Total', ascending=False).head(10)

    fig = px.bar(product, x='Description', y='Total',
                 title="🏆 Top Products", color='Total')

    chart = fig.to_html(full_html=False)

    return render_template('products.html', product_chart=chart)


# 👤 CUSTOMERS DASHBOARD
@app.route('/customers')
def customers():
    global df

    if df is None:
        return "❌ Upload file first!"

    customer = df.groupby('CustomerID')['Total'].sum().reset_index() \
                 .sort_values(by='Total', ascending=False).head(10)

    fig = px.bar(customer, x='CustomerID', y='Total',
                 title="💎 Top Customers", color='Total')

    chart = fig.to_html(full_html=False)

    return render_template('customers.html', customer_chart=chart)


if __name__ == '__main__':
    app.run(debug=True)