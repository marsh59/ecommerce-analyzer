from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

df = None

#  Plotly dark theme 
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


#  Clean dataframe 
def clean_df(raw_df):
    raw_df = raw_df.dropna(subset=['CustomerID'])
    raw_df = raw_df[raw_df['Quantity'] > 0]
    raw_df['CustomerID']  = raw_df['CustomerID'].astype(str).str.replace('.0', '', regex=False)
    raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'], format='mixed')
    raw_df['Total']       = raw_df['Quantity'] * raw_df['UnitPrice']
    return raw_df


#  Home 
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/download-dataset')
def download_dataset():
    return send_from_directory(
        directory=os.path.join(app.root_path),
        path='online_retail.csv',
        as_attachment=True
    )


#  Upload CSV 
@app.route('/upload', methods=['POST'])
def upload():
    global df

    file = request.files.get('file')
    if not file:
        return jsonify({"message": "⚠ No file received."}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        raw = pd.read_csv(path, encoding='latin1')
        df  = clean_df(raw)
        return jsonify({"message": f"✅ CSV uploaded! {len(df):,} rows loaded."})

    except Exception as e:
        return jsonify({"message": f"✖ Error: {str(e)}"}), 500


#  Connect MySQL 
@app.route('/connect_mysql', methods=['POST'])
def connect_mysql():
    global df

    data     = request.get_json()
    host     = data.get('host',     'localhost')
    port     = int(data.get('port', 3306))
    user     = data.get('user',     '')
    password = data.get('password', '')
    database = data.get('database', '')
    table    = data.get('table',    '')

    if not all([user, database, table]):
        return jsonify({"message": "⚠ Please fill in User, Database and Table fields."}), 400

    try:
        import pymysql
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connect_timeout=10
        )
        raw = pd.read_sql(f"SELECT * FROM `{table}`", connection)
        connection.close()

        df = clean_df(raw)
        return jsonify({"message": f"✅ MySQL connected! {len(df):,} rows loaded."})

    except Exception as e:
        return jsonify({"message": f"✖ MySQL Error: {str(e)}"}), 500


#  Sales Dashboard 
@app.route('/sales')
def sales():
    global df
    if df is None:
        return render_template('error.html', message="Please upload a dataset first.")

    #  KPIs 
    total_sales     = f"{df['Total'].sum():,.2f}"
    total_orders    = f"{df['InvoiceNo'].nunique():,}"
    avg_order_value = f"{df['Total'].sum() / df['InvoiceNo'].nunique():,.2f}"

    df['MonthYear'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    monthly         = df.groupby('MonthYear')['Total'].sum()
    best_month      = pd.to_datetime(monthly.idxmax()).strftime('%b %Y')

    #  Trend chart data 
    trend = df.groupby('MonthYear')['Total'].sum().reset_index()
    trend['MonthLabel'] = pd.to_datetime(trend['MonthYear']).dt.strftime('%b %Y')

    # Summary values for trend chart
    peak_idx          = trend['Total'].idxmax()
    low_idx           = trend['Total'].idxmin()
    peak_month_label  = trend.loc[peak_idx, 'MonthLabel']
    peak_month_rev    = f"{trend.loc[peak_idx, 'Total']:,.0f}"
    low_month_label   = trend.loc[low_idx,  'MonthLabel']
    low_month_rev     = f"{trend.loc[low_idx,  'Total']:,.0f}"
    date_start        = trend['MonthLabel'].iloc[0]
    date_end          = trend['MonthLabel'].iloc[-1]

    fig1 = px.line(trend, x='MonthLabel', y='Total',
                   title='Monthly Sales Trend',
                   color_discrete_sequence=['#38bdf8'], markers=True)
    fig1.update_traces(line_width=2.5, fill='tozeroy',
                       fillcolor='rgba(56,189,248,.07)',
                       marker=dict(size=9, color='#38bdf8',
                                   line=dict(width=2, color='#0ea5e9')))
    fig1.update_xaxes(title_text='Month', tickangle=-45,
                      categoryorder='array',
                      categoryarray=trend['MonthLabel'].tolist())
    fig1.update_yaxes(title_text='Revenue (£)', tickprefix='£')

    #  Country chart data 
    country = (df.groupby('Country')['Total'].sum()
                 .reset_index()
                 .sort_values('Total', ascending=False))
    country = country[country['Country'] != 'United Kingdom'].head(10)

    # Summary values for country chart
    top_country_name = country.iloc[0]['Country'] if len(country) > 0 else 'N/A'
    top_country_rev  = f"{country.iloc[0]['Total']:,.0f}" if len(country) > 0 else 'N/A'
    second_country   = country.iloc[1]['Country'] if len(country) > 1 else 'N/A'

    fig2 = px.bar(country, x='Country', y='Total',
                  title='Top 10 Countries by Revenue (Excluding UK)',
                  color='Total', color_continuous_scale=['#1e3a5f', '#38bdf8'])
    fig2.update_layout(coloraxis_showscale=False)
    fig2.update_xaxes(title_text='Country', tickangle=-45)
    fig2.update_yaxes(title_text='Total Revenue (£)', tickprefix='£')

    return render_template('sales.html',
                           total_sales=total_sales,
                           total_orders=total_orders,
                           avg_order_value=avg_order_value,
                           best_month=best_month,
                           peak_month_label=peak_month_label,
                           peak_month_rev=peak_month_rev,
                           low_month_label=low_month_label,
                           low_month_rev=low_month_rev,
                           date_start=date_start,
                           date_end=date_end,
                           top_country_name=top_country_name,
                           top_country_rev=top_country_rev,
                           second_country=second_country,
                           trend_chart=themed_fig(fig1),
                           country_chart=themed_fig(fig2))


# Products Dashboard
@app.route('/products')
def products():
    global df
    if df is None:
        return render_template('error.html', message="Please upload a dataset first.")

    # KPIs 
    total_products = f"{df['Description'].nunique():,}"
    best_product   = df.groupby('Description')['Total'].sum().idxmax()
    best_product_rev = f"{df.groupby('Description')['Total'].sum().max():,.0f}"

    # Product chart
    product = (df.groupby('Description')['Total'].sum()
                 .reset_index()
                 .sort_values('Total', ascending=False)
                 .head(10))

    top_product_name = product.iloc[0]['Description'] if len(product) > 0 else 'N/A'
    second_product   = product.iloc[1]['Description'] if len(product) > 1 else 'N/A'

    fig1 = px.bar(product, x='Total', y='Description', orientation='h',
                  title='Top 10 Products by Revenue',
                  color='Total', color_continuous_scale=['#2d1b69', '#818cf8'])
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'},
                       coloraxis_showscale=False)
    fig1.update_xaxes(title_text='Total Revenue (£)', tickprefix='£')
    fig1.update_yaxes(title_text='Product Name')

    # Category chart
    df['Category'] = df['Description'].str.split().str[0].str.title()
    cat = (df.groupby('Category')['Total'].sum()
             .reset_index()
             .sort_values('Total', ascending=False)
             .head(8))

    top_category_name = cat.iloc[0]['Category'] if len(cat) > 0 else 'N/A'
    top_category_rev  = f"{cat.iloc[0]['Total']:,.0f}" if len(cat) > 0 else 'N/A'
    top_category_pct  = f"{(cat.iloc[0]['Total'] / cat['Total'].sum() * 100):.1f}" if len(cat) > 0 else 'N/A'

    fig2 = px.pie(cat, names='Category', values='Total',
                  title='Revenue by Product Category',
                  color_discrete_sequence=ACCENT_COLORS, hole=0.45)
    fig2.update_traces(textposition='outside')

    return render_template('products.html',
                           total_products=total_products,
                           best_product=best_product,
                           best_product_rev=best_product_rev,
                           top_product_name=top_product_name,
                           second_product=second_product,
                           top_category_name=top_category_name,
                           top_category_rev=top_category_rev,
                           top_category_pct=top_category_pct,
                           product_chart=themed_fig(fig1),
                           category_chart=themed_fig(fig2))


# Customers Dashboard
@app.route('/customers')
def customers():
    global df
    if df is None:
        return render_template('error.html', message="Please upload a dataset first.")

    # KPIs
    total_customers = f"{df['CustomerID'].nunique():,}"
    most_loyal      = df.groupby('CustomerID')['InvoiceNo'].nunique().idxmax()
    most_loyal_orders = int(df.groupby('CustomerID')['InvoiceNo'].nunique().max())

    # Top customers by revenue
    customer = (df.groupby('CustomerID')['Total'].sum()
                  .reset_index()
                  .sort_values('Total', ascending=False)
                  .head(10))

    top_customer_id  = customer.iloc[0]['CustomerID'] if len(customer) > 0 else 'N/A'
    top_customer_rev = f"{customer.iloc[0]['Total']:,.0f}" if len(customer) > 0 else 'N/A'

    fig1 = px.bar(customer, x='CustomerID', y='Total',
                  title='Top 10 Customers by Revenue',
                  color='Total', color_continuous_scale=['#431407', '#fb923c'])
    fig1.update_layout(coloraxis_showscale=False)
    fig1.update_xaxes(title_text='Customer ID', tickangle=-45)
    fig1.update_yaxes(title_text='Total Revenue (£)', tickprefix='£')

    # Repeat orders
    repeat = (df.groupby('CustomerID')['InvoiceNo'].nunique()
                .reset_index()
                .rename(columns={'InvoiceNo': 'Orders'})
                .sort_values('Orders', ascending=False)
                .head(10))

    top_repeat_id     = repeat.iloc[0]['CustomerID'] if len(repeat) > 0 else 'N/A'
    top_repeat_orders = int(repeat.iloc[0]['Orders']) if len(repeat) > 0 else 0

    fig2 = px.bar(repeat, x='CustomerID', y='Orders',
                  title='Top 10 Customers by Order Count',
                  color='Orders', color_continuous_scale=['#052e16', '#34d399'])
    fig2.update_layout(coloraxis_showscale=False)
    fig2.update_xaxes(title_text='Customer ID', tickangle=-45)
    fig2.update_yaxes(title_text='Number of Orders')

    # Customers by country
    customer_country = (df.groupby('Country')['CustomerID'].nunique()
                          .reset_index()
                          .rename(columns={'CustomerID': 'Customers'})
                          .sort_values('Customers', ascending=False))
    customer_country = customer_country[customer_country['Country'] != 'United Kingdom'].head(10)

    top_country_customers      = customer_country.iloc[0]['Country'] if len(customer_country) > 0 else 'N/A'
    top_country_customers_count = int(customer_country.iloc[0]['Customers']) if len(customer_country) > 0 else 0

    fig3 = px.pie(customer_country, names='Country', values='Customers',
                  title='Top 10 Countries by Customers (Excluding UK)',
                  color_discrete_sequence=ACCENT_COLORS, hole=0.45)
    fig3.update_traces(textposition='outside', textinfo='percent+label')

    return render_template('customers.html',
                           total_customers=total_customers,
                           most_loyal=most_loyal,
                           most_loyal_orders=most_loyal_orders,
                           top_customer_id=top_customer_id,
                           top_customer_rev=top_customer_rev,
                           top_repeat_id=top_repeat_id,
                           top_repeat_orders=top_repeat_orders,
                           top_country_customers=top_country_customers,
                           top_country_customers_count=top_country_customers_count,
                           customer_chart=themed_fig(fig1),
                           repeat_chart=themed_fig(fig2),
                           country_chart=themed_fig(fig3))


# Forecasting
@app.route('/forecast')
def forecast():
    global df
    if df is None:
        return render_template('error.html', message="Please upload a dataset first.")

    df['MonthYear'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    monthly = df.groupby('MonthYear')['Total'].sum().reset_index()
    monthly.columns = ['ds', 'y']
    monthly['ds'] = pd.to_datetime(monthly['ds'])
    monthly = monthly.sort_values('ds')

    model = ExponentialSmoothing(
        monthly['y'],
        trend='add',
        seasonal=None,
        initialization_method='estimated'
    )
    fit = model.fit()

    forecast_vals = fit.forecast(6)
    future_dates  = pd.date_range(
        start=monthly['ds'].iloc[-1] + pd.DateOffset(months=1),
        periods=6, freq='MS'
    )

    upper = forecast_vals * 1.15
    lower = forecast_vals * 0.85

    last_actual_rev   = monthly['y'].iloc[-1]
    last_actual_month = monthly['ds'].iloc[-1].strftime('%b %Y')
    last_actual_fmt   = f"{last_actual_rev:,.0f}"
    first_pred_month  = future_dates[0].strftime('%b %Y')
    last_pred_month   = future_dates[-1].strftime('%b %Y')
    avg_pred_rev      = f"{forecast_vals.mean():,.0f}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['ds'], y=monthly['y'],
        name='Actual Sales',
        line=dict(color='#38bdf8', width=2.5),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast_vals,
        name='Predicted Sales',
        line=dict(color='#fb923c', width=2.5, dash='dot'),
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill='toself', fillcolor='rgba(251,146,60,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Range'
    ))

    fig.update_layout(
        title='Sales Forecast (Next 6 Months)',
        xaxis_title='Month', yaxis_title='Revenue (£)',
        **PLOTLY_LAYOUT
    )
    fig.update_yaxes(tickprefix='£')

    return render_template('forecast.html',
                           forecast_chart=fig.to_html(full_html=False,
                                                      config={'displayModeBar': False}),
                           first_pred_month=first_pred_month,
                           last_pred_month=last_pred_month,
                           avg_pred_rev=avg_pred_rev,
                           last_actual_month=last_actual_month,
                           last_actual_fmt=last_actual_fmt)


# RFM Analysis 
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

    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'],   q=4, labels=[4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'),  q=4, labels=[1, 2, 3, 4])
    rfm_df['RFM_Score'] = (rfm_df['R_Score'].astype(str) +
                            rfm_df['F_Score'].astype(str) +
                            rfm_df['M_Score'].astype(str))

    def assign_segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        if r >= 4 and f >= 4 and m >= 4: return 'Champion'
        elif r >= 3 and f >= 3:           return 'Loyal'
        elif r >= 3 and f <= 2:           return 'New Customer'
        elif r <= 2 and f >= 3:           return 'At Risk'
        else:                             return 'Lost'

    rfm_df['Segment'] = rfm_df.apply(assign_segment, axis=1)

    total_champions    = len(rfm_df[rfm_df['Segment'] == 'Champion'])
    total_loyal        = len(rfm_df[rfm_df['Segment'] == 'Loyal'])
    total_at_risk      = len(rfm_df[rfm_df['Segment'] == 'At Risk'])
    total_lost         = len(rfm_df[rfm_df['Segment'] == 'Lost'])
    total_new_customer = len(rfm_df[rfm_df['Segment'] == 'New Customer'])
    total_customers_rfm = len(rfm_df)

    # Average monetary per segment for summary
    champ_avg  = f"{rfm_df[rfm_df['Segment']=='Champion']['Monetary'].mean():,.0f}" if total_champions > 0 else '0'
    loyal_avg  = f"{rfm_df[rfm_df['Segment']=='Loyal']['Monetary'].mean():,.0f}"    if total_loyal > 0    else '0'
    risk_avg   = f"{rfm_df[rfm_df['Segment']=='At Risk']['Monetary'].mean():,.0f}"  if total_at_risk > 0  else '0'

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
    fig2.update_xaxes(title_text='Customer Segment')
    fig2.update_yaxes(title_text='Average Revenue (£)', tickprefix='£')

    return render_template('rfm.html',
                           total_champions=total_champions,
                           total_loyal=total_loyal,
                           total_at_risk=total_at_risk,
                           total_lost=total_lost,
                           total_new_customer=total_new_customer,
                           total_customers_rfm=total_customers_rfm,
                           champ_avg=champ_avg,
                           loyal_avg=loyal_avg,
                           risk_avg=risk_avg,
                           segment_chart=themed_fig(fig1),
                           monetary_chart=themed_fig(fig2))


#  Run 
if __name__ == '__main__':
    app.run(debug=False)