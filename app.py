import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Yes Bank Stock Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B1220; color: #ffffff; }
    .main-header {
        background: linear-gradient(135deg, #1A2333, #0B1220);
        border: 1px solid rgba(251,191,36,0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-title {
        color: #FBB024;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(251,191,36,0.3);
    }
    div[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(251,191,36,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data_YesBank_StockPrices.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
    df = df.sort_values('Date').reset_index(drop=True)
    df['Returns'] = df['Close'].pct_change() * 100
    df['MA_3'] = df['Close'].rolling(window=3).mean()
    df['MA_6'] = df['Close'].rolling(window=6).mean()
    df['MA_12'] = df['Close'].rolling(window=12).mean()
    df['Volatility'] = df['Returns'].rolling(window=6).std()
    df['Price_Range'] = df['High'] - df['Low']
    return df

df = load_data()

with st.sidebar:
    st.markdown("### 🏦 Yes Bank Stock")
    st.markdown("---")
    page = st.radio("📊 Select Page", [
        "🏠 Overview",
        "📈 Price Analysis",
        "🤖 ML Prediction",
        "📊 Technical Analysis",
        "📋 Data Explorer"
    ])
    st.markdown("---")
    st.metric("📅 Data Period", f"{df['Date'].dt.year.min()} - {df['Date'].dt.year.max()}")
    st.metric("📊 Total Records", f"{len(df):,}")

st.markdown("""
<div class="main-header">
    <h1 style="color:#FBB024; margin:0; font-size:2rem;">🏦 Yes Bank Stock Price Prediction</h1>
    <p style="color:#94A3B8; margin:0.5rem 0 0 0;">ML-Powered Financial Forecasting | Linear Regression & Time Series Analysis</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("💰 Latest Close", f"₹{df['Close'].iloc[-1]:.2f}")
with col2:
    st.metric("📈 All-Time High", f"₹{df['High'].max():.2f}")
with col3:
    st.metric("📉 All-Time Low", f"₹{df['Low'].min():.2f}")
with col4:
    avg_return = df['Returns'].mean()
    st.metric("📊 Avg Monthly Return", f"{avg_return:.2f}%")
with col5:
    volatility = df['Returns'].std()
    st.metric("⚡ Volatility", f"{volatility:.2f}%")

st.markdown("---")

if page == "🏠 Overview":
    st.markdown('<p class="section-title">📈 Stock Price History</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='OHLC',
        increasing_line_color='#22C55E',
        decreasing_line_color='#EF4444'
    ))
    fig.update_layout(
        paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'), height=400,
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
            name='Close', line=dict(color='#FBB024', width=2)))
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['MA_3'],
            name='MA 3M', line=dict(color='#3B82F6', width=1.5, dash='dot')))
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['MA_6'],
            name='MA 6M', line=dict(color='#8B5CF6', width=1.5, dash='dot')))
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['MA_12'],
            name='MA 12M', line=dict(color='#EF4444', width=1.5, dash='dot')))
        fig2.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='Close Price with Moving Averages',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=df['Date'], y=df['Returns'],
            marker_color=df['Returns'].apply(lambda x: '#22C55E' if x > 0 else '#EF4444'),
            name='Monthly Returns'))
        fig3.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='Monthly Returns (%)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig3, use_container_width=True)

elif page == "📈 Price Analysis":
    st.markdown('<p class="section-title">📈 Price Deep Dive</p>', unsafe_allow_html=True)
    
    yearly = df.groupby(df['Date'].dt.year).agg({
        'Open': 'first', 'Close': 'last',
        'High': 'max', 'Low': 'min'
    }).reset_index()
    yearly.columns = ['Year', 'Open', 'Close', 'High', 'Low']
    
    fig = px.bar(yearly, x='Year', y=['High', 'Low'],
        barmode='group', title='Yearly High & Low Prices',
        color_discrete_map={'High': '#22C55E', 'Low': '#EF4444'})
    fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(df, x='Returns', nbins=30,
            title='Return Distribution',
            color_discrete_sequence=['#FBB024'])
        fig2.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['Date'], y=df['Volatility'],
            fill='tozeroy', line=dict(color='#F97316', width=2),
            fillcolor='rgba(249,115,22,0.1)', name='Volatility'))
        fig3.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='Rolling Volatility (6M)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig3, use_container_width=True)

elif page == "🤖 ML Prediction":
    st.markdown('<p class="section-title">🤖 Stock Price Prediction Model</p>', unsafe_allow_html=True)
    
    df_ml = df.dropna().copy()
    df_ml['Month'] = df_ml['Date'].dt.month
    df_ml['Year_Num'] = df_ml['Date'].dt.year
    
    features = ['Open', 'High', 'Low', 'MA_3', 'MA_6', 'Month', 'Year_Num', 'Price_Range']
    X = df_ml[features]
    y = df_ml['Close']
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 R² Score", f"{r2:.4f}")
    with col2:
        st.metric("📉 MAE", f"₹{mae:.2f}")
    with col3:
        st.metric("✅ Accuracy", f"{r2*100:.1f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ml['Date'].iloc[split:], y=y_test,
        name='Actual', line=dict(color='#FBB024', width=2)))
    fig.add_trace(go.Scatter(
        x=df_ml['Date'].iloc[split:], y=y_pred,
        name='Predicted', line=dict(color='#3B82F6', width=2, dash='dot')))
    fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'), title='Actual vs Predicted Stock Price',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<p class="section-title">🎯 Predict Future Price</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        open_price = st.number_input("📈 Open Price (₹)", 0.0, 500.0, float(df['Open'].iloc[-1]))
        high_price = st.number_input("📈 High Price (₹)", 0.0, 500.0, float(df['High'].iloc[-1]))
    with col2:
        low_price = st.number_input("📉 Low Price (₹)", 0.0, 500.0, float(df['Low'].iloc[-1]))
        ma3 = st.number_input("📊 MA 3M (₹)", 0.0, 500.0, float(df['MA_3'].iloc[-1]))
    with col3:
        ma6 = st.number_input("📊 MA 6M (₹)", 0.0, 500.0, float(df['MA_6'].iloc[-1]))
        month = st.slider("📅 Month", 1, 12, 6)
        year = st.slider("📅 Year", 2020, 2026, 2025)
    
    price_range = high_price - low_price
    input_data = np.array([[open_price, high_price, low_price, ma3, ma6, month, year, price_range]])
    predicted_price = model.predict(input_data)[0]
    
    st.markdown(f"""
    <div style="background:#1A2333; border:2px solid #FBB024; border-radius:16px; 
    padding:2rem; text-align:center; margin-top:1rem;">
        <h2 style="color:#FBB024; margin:0;">🏦 Predicted Close Price: ₹{predicted_price:.2f}</h2>
        <p style="color:#94A3B8; margin:0.5rem 0 0 0;">Based on Linear Regression Model</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "📊 Technical Analysis":
    st.markdown('<p class="section-title">📊 Technical Indicators</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
        name='Close Price', line=dict(color='#FBB024', width=2)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_3'],
        name='MA 3M', line=dict(color='#3B82F6', width=1.5)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_6'],
        name='MA 6M', line=dict(color='#8B5CF6', width=1.5)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_12'],
        name='MA 12M', line=dict(color='#EF4444', width=1.5)))
    fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'), height=400, title='Moving Averages',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Price_Range'],
            fill='tozeroy', line=dict(color='#22C55E', width=2),
            fillcolor='rgba(34,197,94,0.1)', name='Price Range'))
        fig2.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='Daily Price Range (High-Low)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        corr = df[['Open','High','Low','Close','Returns','Volatility']].corr()
        fig3 = px.imshow(corr, color_continuous_scale='YlOrRd',
            title='Feature Correlation Matrix')
        fig3.update_layout(paper_bgcolor='#1A2333', font=dict(color='#94A3B8'))
        st.plotly_chart(fig3, use_container_width=True)

elif page == "📋 Data Explorer":
    st.markdown('<p class="section-title">📋 Raw Data Explorer</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        year_filter = st.multiselect("📅 Filter by Year",
            options=sorted(df['Date'].dt.year.unique()),
            default=sorted(df['Date'].dt.year.unique()))
    with col2:
        show_cols = st.multiselect("📊 Select Columns",
            options=['Date','Open','High','Low','Close','Returns','MA_3','MA_6','MA_12','Volatility'],
            default=['Date','Open','High','Low','Close','Returns'])
    
    filtered = df[df['Date'].dt.year.isin(year_filter)][show_cols]
    st.dataframe(filtered, use_container_width=True, height=400)
    csv = filtered.to_csv(index=False)
    st.download_button("📥 Download Data", csv, "yes_bank_stock.csv", "text/csv")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.8rem; padding:1rem;">
    🏦 Yes Bank Stock Prediction | Built by <strong style="color:#FBB024">Taiyba Shaikh</strong> | ML Project
</div>
""", unsafe_allow_html=True)