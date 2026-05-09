from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import plotly.express as px
from prophet import Prophet
import plotly.graph_objects as go

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

df = None

# ─── Plotly dark theme ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(14,21,36,0.6)',
    font=dict(color='#94a3b8', family='DM Sans, sans-serif', size=12),
    title_font=dict(color='#e2e8f0', size=15, family='Space Grotesk, sans-serif'),
    xaxis=dict(
        gridcolor='rgba(100,160,255,.08)',
        zerolinecolor='rgba(100,160,255,.08)',
        linecolor='rgba(100,160,255,.12)'
    ),
    yaxis=dict(
        gridcolor='rgba(100,160,255,.08)',
        zerolinecolor='rgba(100,160,255,.08)',
        linecolor='rgba(100,160,255,.12)'
    ),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(100,160,255,.15)'),
    margin=dict(t=50, l=10, r=10, b=80),
)

ACCENT_COLORS = ['#38bdf8', '#818cf8', '#fb923c', '#34d399', '#f472b6', '#a78bfa', '#fbbf24']


def themed_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig.to_html(full_html=False, config={'displayModeBar': False})


# ─── Helper: clean dataframe (same logic for both CSV and MySQL) ────────────
def clean_df(raw_df):
    raw_df = raw_df.dropna(subset=['CustomerID'])
    raw_df = raw_df[raw_df['Quantity'] > 0]
    raw_df['CustomerID']  = raw_df['CustomerID'].astype(str).str.replace('.0', '', regex=False)
    raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'], format='mixed')
    raw_df['Total']       = raw_df['Quantity'] * raw_df['UnitPrice']
    return raw_df


# ─── Home ──────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


# ─── Upload CSV ────────────────────────────────────────────────────────────
@app.route('/upload', methods=['POST'])
def upload():
    global df

    file = request.files.get('file')
    if not file:
        return jsonify({"message": "⚠ No file received."}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        # ── CSV path ──────────────────────────────────────────────────────
        raw = pd.read_csv(path, encoding='latin1')
        df  = clean_df(raw)
        return jsonify({"message": f"✅ CSV uploaded! {len(df):,} rows loaded."})

    except Exception as e:
        return jsonify({"message": f"✖ Error: {str(e)}"}), 500


# ─── Connect MySQL ─────────────────────────────────────────────────────────
@app.route('/connect_mysql', methods=['POST'])
def connect_mysql():
    global df

    # Read connection details sent from the form
    data      = request.get_json()
    host      = data.get('host',     'localhost')
    port      = int(data.get('port', 3306))
    user      = data.get('user',     '')
    password  = data.get('password', '')
    database  = data.get('database', '')
    table     = data.get('table',    '')

    if not all([user, database, table]):
        return jsonify({"message": "⚠ Please fill in all MySQL fields."}), 400

    try:
        # ── MySQL path ────────────────────────────────────────────────────
        # pip install pymysql   (run once in your terminal)
        import pymysql
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        raw = pd.read_sql(f"SELECT * FROM `{table}`", connection)
        connection.close()

        df = clean_df(raw)
        return jsonify({"message": f"✅ MySQL connected! {len(df):,} rows loaded."})

    except Exception as e:
        return jsonify({"message": f"✖ MySQL Error: {str(e)}"}), 500


# ─── Sales Dashboard ───────────────────────────────────────────────────────
@app.route('/sales')
def sales():
    global df
    if df is None:
        return render_template('error.html', message="Please upload a dataset first.")

    total_sales     = f"{df['Total'].sum():,.2f}"
    total_orders    = f"{df['InvoiceNo'].nunique():,}"
    avg_order_value = f"{df['Total'].sum() / df['InvoiceNo'].nunique():,.2f}"

    df['MonthYear'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    monthly    = df.groupby('MonthYear')['Total'].sum()
    best_month = pd.to_datetime(monthly.idxmax()).strftime('%b %Y')

    trend = df.groupby('MonthYear')['Total'].sum().reset_index()
    trend['MonthLabel'] = pd.to_datetime(trend['MonthYear']).dt.strftime('%b %Y')

    fig1 = px.line(trend, x='MonthLabel', y='Total',
                   title='Monthly Sales Trend',
                   color_discrete_sequence=['#38bdf8'], markers=True)
    fig1.update_traces(line_width=2.5, fill='tozeroy',
                       fillcolor='rgba(56,189,248,.07)',
                       marker=dict(size=9, color='#38bdf8',
                                   line=dict(width=2, color='#0ea5e9')))
    fig1.update_xaxes(tickangle=-45, categoryorder='array',
                      categoryarray=trend['MonthLabel'].tolist())
    fig1.update_yaxes(tickprefix='£')

    country = (df.groupby('Country')['Total'].sum()
                 .reset_index()
                 .sort_values('Total', ascending=False))
    country = country[country['Country'] != 'United Kingdom'].head(10)

    fig2 = px.bar(country, x='Country', y='Total',
                  title='Top 10 Countries by Revenue (Excluding UK)',
                  color='Total', color_continuous_scale=['#1e3a5f', '#38bdf8'])
    fig2.update_layout(coloraxis_showscale=False)
    fig2.update_xaxes(tickangle=-45)
    fig2.update_yaxes(tickprefix='£')

    return render_template('sales.html',
                           total_sales=total_sales,
                           total_orders=total_orders,
                           avg_order_value=avg_order_value,
                           best_month=best_month,
                           trend_chart=themed_fig(fig1),
                           country_chart=themed_fig(fig2))


# ─── Products Dashboard ────────────────────────────────────────────────────
@app.route('/products')
def products():
    global df
    if df is None:
        return render_template('error.html', message="Please upload a dataset first.")

    total_products = f"{df['Description'].nunique():,}"
    best_product   = df.groupby('Description')['Total'].sum().idxmax()

    product = (df.groupby('Description')['Total'].sum()
                 .reset_index()
                 .sort_values('Total', ascending=False)
                 .head(10))

    fig1 = px.bar(product, x='Total', y='Description', orientation='h',
                  title='Top 10 Products by Revenue',
                  color='Total', color_continuous_scale=['#2d1b69', '#818cf8'])
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'},
                       coloraxis_showscale=False)
    fig1.update_xaxes(tickprefix='£')

    df['Category'] = df['Description'].str.split().str[0].str.title()
    cat = (df.groupby('Category')['Total'].sum()
             .reset_index()
             .sort_values('Total', ascending=False)
             .head(8))

    fig2 = px.pie(cat, names='Category', values='Total',
                  title='Revenue by Product Category',
                  color_discrete_sequence=ACCENT_COLORS, hole=0.45)
    fig2.update_traces(textposition='outside')

    return render_template('products.html',
                           total_products=total_products,
                           best_product=best_product,
                           product_chart=themed_fig(fig1),
                           category_chart=themed_fig(fig2))


# ─── Customers Dashboard ───────────────────────────────────────────────────
@app.route('/customers')
def customers():
    global df
    if df is None:
        return render_template('error.html', message="Please upload a dataset first.")

    total_customers = f"{df['CustomerID'].nunique():,}"
    most_loyal      = df.groupby('CustomerID')['InvoiceNo'].nunique().idxmax()

    customer = (df.groupby('CustomerID')['Total'].sum()
                  .reset_index()
                  .sort_values('Total', ascending=False)
                  .head(10))

    fig1 = px.bar(customer, x='CustomerID', y='Total',
                  title='Top 10 Customers by Revenue',
                  color='Total', color_continuous_scale=['#431407', '#fb923c'])
    fig1.update_layout(coloraxis_showscale=False)
    fig1.update_xaxes(tickangle=-45)
    fig1.update_yaxes(tickprefix='£')

    repeat = (df.groupby('CustomerID')['InvoiceNo'].nunique()
                .reset_index()
                .rename(columns={'InvoiceNo': 'Orders'})
                .sort_values('Orders', ascending=False)
                .head(10))

    fig2 = px.bar(repeat, x='CustomerID', y='Orders',
                  title='Top 10 Customers by Order Count',
                  color='Orders', color_continuous_scale=['#052e16', '#34d399'])
    fig2.update_layout(coloraxis_showscale=False)
    fig2.update_xaxes(tickangle=-45)

    customer_country = (df.groupby('Country')['CustomerID'].nunique()
                          .reset_index()
                          .rename(columns={'CustomerID': 'Customers'})
                          .sort_values('Customers', ascending=False))
    customer_country = customer_country[customer_country['Country'] != 'United Kingdom'].head(10)

    fig3 = px.pie(customer_country, names='Country', values='Customers',
                  title='Top 10 Countries by Customers (Excluding UK)',
                  color_discrete_sequence=ACCENT_COLORS, hole=0.45)
    fig3.update_traces(textposition='outside', textinfo='percent+label')

    return render_template('customers.html',
                           total_customers=total_customers,
                           most_loyal=most_loyal,
                           customer_chart=themed_fig(fig1),
                           repeat_chart=themed_fig(fig2),
                           country_chart=themed_fig(fig3))


# ─── Forecasting ───────────────────────────────────────────────────────────
@app.route('/forecast')
def forecast():
    global df
    if df is None:
        return render_template('error.html', message="Please upload a dataset first.")

    df['MonthYear'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    monthly = df.groupby('MonthYear')['Total'].sum().reset_index()
    monthly.columns = ['ds', 'y']
    monthly['ds'] = pd.to_datetime(monthly['ds'])

    model = Prophet(yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False)
    model.fit(monthly)

    future       = model.make_future_dataframe(periods=6, freq='ME')
    forecast_df  = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['ds'], y=monthly['y'],
                             name='Actual Sales',
                             line=dict(color='#38bdf8', width=2.5),
                             mode='lines+markers',
                             marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'],
                             name='Predicted Sales',
                             line=dict(color='#fb923c', width=2.5, dash='dot'),
                             mode='lines'))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
        y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(251,146,60,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Range'))

    fig.update_layout(title='Sales Forecast (Next 6 Months)',
                      xaxis_title='Month', yaxis_title='Revenue (£)',
                      **PLOTLY_LAYOUT)
    fig.update_yaxes(tickprefix='£')

    return render_template('forecast.html',
                           forecast_chart=fig.to_html(full_html=False,
                                                      config={'displayModeBar': False}))


# ─── RFM Analysis ──────────────────────────────────────────────────────────
@app.route('/rfm')
def rfm():
    global df
    if df is None:
        return render_template('error.html', message="Please upload a dataset first.")

    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm_df = df.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency = ('InvoiceNo',   'nunique'),
        Monetary  = ('Total',       'sum')
    ).reset_index()

    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'],   q=4, labels=[4,3,2,1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=4, labels=[1,2,3,4])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'),  q=4, labels=[1,2,3,4])
    rfm_df['RFM_Score'] = (rfm_df['R_Score'].astype(str) +
                           rfm_df['F_Score'].astype(str) +
                           rfm_df['M_Score'].astype(str))

    def assign_segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        if r >= 4 and f >= 4 and m >= 4:  return 'Champion'
        elif r >= 3 and f >= 3:            return 'Loyal'
        elif r >= 3 and f <= 2:            return 'New Customer'
        elif r <= 2 and f >= 3:            return 'At Risk'
        else:                              return 'Lost'

    rfm_df['Segment'] = rfm_df.apply(assign_segment, axis=1)

    total_champions = len(rfm_df[rfm_df['Segment'] == 'Champion'])
    total_loyal     = len(rfm_df[rfm_df['Segment'] == 'Loyal'])
    total_at_risk   = len(rfm_df[rfm_df['Segment'] == 'At Risk'])
    total_lost      = len(rfm_df[rfm_df['Segment'] == 'Lost'])

    segment_counts = rfm_df['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']

    fig1 = px.pie(segment_counts, names='Segment', values='Count',
                  title='Customer Segments',
                  color_discrete_sequence=ACCENT_COLORS, hole=0.45)
    fig1.update_traces(textposition='outside', textinfo='percent+label')

    seg_monetary = rfm_df.groupby('Segment')['Monetary'].mean().reset_index()
    seg_monetary.columns = ['Segment', 'Avg Revenue']

    fig2 = px.bar(seg_monetary, x='Segment', y='Avg Revenue',
                  title='Average Revenue per Customer Segment',
                  color='Avg Revenue', color_continuous_scale=['#1e3a5f', '#38bdf8'])
    fig2.update_layout(coloraxis_showscale=False)
    fig2.update_yaxes(tickprefix='£')

    return render_template('rfm.html',
                           total_champions=total_champions,
                           total_loyal=total_loyal,
                           total_at_risk=total_at_risk,
                           total_lost=total_lost,
                           segment_chart=themed_fig(fig1),
                           monetary_chart=themed_fig(fig2))


# ─── Run ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)