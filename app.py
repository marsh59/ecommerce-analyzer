from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

df = None

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global df

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath, encoding='latin1')

    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['Total'] = df['Quantity'] * df['UnitPrice']

    return jsonify({"message": "File uploaded successfully!"})


# 📊 SALES
@app.route('/sales')
def sales():
    global df

    if df is None:
        return jsonify({"error": "Upload file first!"})

    total_sales = df['Total'].sum()
    total_orders = df['InvoiceNo'].nunique()

    sales_data = df.groupby('InvoiceDate')['Total'].sum().head(15)

    plt.figure(figsize=(10,5))
    sales_data.plot(marker='o')
    plt.xticks(rotation=45)
    plt.xlabel("Invoice Date")
    plt.ylabel("Sales")
    plt.title("Sales Trend")
    plt.tight_layout()
    plt.savefig('static/sales_chart.png')
    plt.close()

    return jsonify({
        "total_sales": round(total_sales,2),
        "total_orders": int(total_orders),
        "chart": "/static/sales_chart.png"
    })


# 📦 PRODUCTS
@app.route('/products')
def products():
    global df

    if df is None:
        return jsonify({"error": "Upload file first!"})

    product_sales = df.groupby('Description')['Total'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,5))
    product_sales.plot(kind='bar')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel("Products")
    plt.ylabel("Sales")
    plt.title("Top Products")
    plt.tight_layout()
    plt.savefig('static/product_chart.png')
    plt.close()

    return jsonify({
        "products": product_sales.to_dict(),
        "chart": "/static/product_chart.png"
    })


# 👤 CUSTOMERS
@app.route('/customers')
def customers():
    global df

    if df is None:
        return jsonify({"error": "Upload file first!"})

    customer_data = df.groupby('CustomerID')['Total'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,5))
    customer_data.plot(kind='bar')
    plt.xticks(rotation=30)
    plt.xlabel("Customer ID")
    plt.ylabel("Total Spending")
    plt.title("Top Customers")
    plt.tight_layout()
    plt.savefig('static/customer_chart.png')
    plt.close()

    return jsonify({
        "customers": customer_data.to_dict(),
        "chart": "/static/customer_chart.png"
    })


if __name__ == '__main__':
    app.run(debug=True)