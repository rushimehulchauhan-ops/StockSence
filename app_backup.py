import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StockSense - Stock Market Analytics",
    layout="wide"
)


# =========================================================
# STOCK LIST
# =========================================================

STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Axis Bank": "AXISBANK.NS"
}


# =========================================================
# ML FEATURES
# =========================================================

ML_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "EMA_50",
    "RSI_14"
]


# =========================================================
# FREE DATA DOWNLOAD
# =========================================================

@st.cache_data(ttl=3600)
def download_stock_data(ticker):

    try:

        data = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if data is None or data.empty:
            return None, "No data returned from Yahoo Finance."

        # Handle yfinance MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):

            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        # Keep only required columns
        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing:
            return None, f"Missing columns: {missing}"

        data = data[required_columns].copy()

        data["Date"] = pd.to_datetime(data["Date"])

        data = data.dropna()

        data = data.sort_values("Date")

        data = data.reset_index(drop=True)

        return data, "Downloaded from Yahoo Finance"

    except Exception as e:

        return None, str(e)


# =========================================================
# CSV FALLBACK
# =========================================================

def load_csv_fallback(ticker):

    file_path = f"data/raw/{ticker.replace('.NS', '')}.csv"

    if not os.path.exists(file_path):

        return None

    try:

        data = pd.read_csv(file_path)

        # Handle old yfinance CSV format
        if "Price" in data.columns:

            data = pd.read_csv(
                file_path,
                skiprows=[1, 2]
            )

            data = data.rename(
                columns={"Price": "Date"}
            )

        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            col in data.columns
            for col in required_columns
        ):

            return None

        data = data[required_columns].copy()

        data["Date"] = pd.to_datetime(
            data["Date"]
        )

        data = data.dropna()

        data = data.sort_values("Date")

        data = data.reset_index(drop=True)

        return data

    except Exception:

        return None


# =========================================================
# LOAD STOCK DATA WITH FALLBACK
# =========================================================

def get_stock_data(ticker):

    data, source = download_stock_data(ticker)

    if data is not None:

        return data, source

    fallback_data = load_csv_fallback(ticker)

    if fallback_data is not None:

        return (
            fallback_data,
            "Local CSV fallback"
        )

    return None, source


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    # Daily return
    df["Daily_Return"] = (
        df["Close"].pct_change() * 100
    )

    # Simple Moving Average
    df["SMA_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["SMA_50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    # Exponential Moving Average
    df["EMA_20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA_50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = avg_gain / avg_loss

    df["RSI_14"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    return df


# =========================================================
# CREATE ML DATA
# =========================================================

def prepare_ml_data(df):

    data = df.copy()

    # Target:
    # 1 = next day's Close is higher
    # 0 = next day's Close is lower/equal

    data["Target"] = (
        data["Close"].shift(-1)
        > data["Close"]
    ).astype(int)

    # Last row has no known future target
    data = data.iloc[:-1].copy()

    data = data.dropna(
        subset=ML_FEATURES + ["Target"]
    )

    return data


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models(df):

    ml_data = prepare_ml_data(df)

    if len(ml_data) < 300:

        return None

    X = ml_data[ML_FEATURES]

    y = ml_data["Target"]

    # Chronological 80/20 split
    split_index = int(
        len(ml_data) * 0.80
    )

    X_train = X.iloc[:split_index]

    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]

    y_test = y.iloc[split_index:]


    # Logistic Regression
    logistic_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])


    logistic_model.fit(
        X_train,
        y_train
    )


    # Random Forest
    random_forest_model = (
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            random_state=42
        )
    )


    random_forest_model.fit(
        X_train,
        y_train
    )


    # Predictions
    logistic_pred = (
        logistic_model.predict(X_test)
    )

    random_forest_pred = (
        random_forest_model.predict(X_test)
    )


    # Metrics
    logistic_metrics = {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(
            y_test,
            logistic_pred
        ),
        "Precision": precision_score(
            y_test,
            logistic_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            logistic_pred,
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_test,
            logistic_pred,
            zero_division=0
        )
    }


    random_forest_metrics = {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(
            y_test,
            random_forest_pred
        ),
        "Precision": precision_score(
            y_test,
            random_forest_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            random_forest_pred,
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_test,
            random_forest_pred,
            zero_division=0
        )
    }


    metrics_df = pd.DataFrame([
        logistic_metrics,
        random_forest_metrics
    ])


    # Prediction history
    prediction_history = pd.DataFrame({
        "Date": ml_data["Date"].iloc[
            split_index:
        ].values,

        "Close": ml_data["Close"].iloc[
            split_index:
        ].values,

        "Actual": y_test.values,

        "Logistic_Prediction":
            logistic_pred,

        "RandomForest_Prediction":
            random_forest_pred
    })


    prediction_history["Actual_Label"] = (
        prediction_history["Actual"]
        .map({
            0: "DOWN",
            1: "UP"
        })
    )


    prediction_history["Logistic_Label"] = (
        prediction_history[
            "Logistic_Prediction"
        ].map({
            0: "DOWN",
            1: "UP"
        })
    )


    prediction_history[
        "RandomForest_Label"
    ] = (
        prediction_history[
            "RandomForest_Prediction"
        ].map({
            0: "DOWN",
            1: "UP"
        })
    )


    return {
        "logistic_model":
            logistic_model,

        "random_forest_model":
            random_forest_model,

        "metrics":
            metrics_df,

        "prediction_history":
            prediction_history,

        "ml_data":
            ml_data,

        "X_test":
            X_test,

        "y_test":
            y_test,

        "logistic_pred":
            logistic_pred,

        "random_forest_pred":
            random_forest_pred
    }


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("StockSense")

selected_stock_name = st.sidebar.selectbox(
    "Select Stock",
    list(STOCKS.keys())
)

selected_ticker = STOCKS[
    selected_stock_name
]


st.sidebar.caption(
    f"Ticker: {selected_ticker}"
)


# =========================================================
# LOAD SELECTED STOCK
# =========================================================

df, data_source = get_stock_data(
    selected_ticker
)


if df is None:

    st.error(
        "Stock data could not be loaded."
    )

    st.info(
        "Internet download failed and no "
        "local CSV fallback was found."
    )

    st.stop()


# =========================================================
# ADD INDICATORS
# =========================================================

df = add_indicators(df)


# =========================================================
# DATA SOURCE STATUS
# =========================================================

if "Yahoo Finance" in data_source:

    st.sidebar.success(
        "Data source: Free Yahoo Finance"
    )

else:

    st.sidebar.warning(
        "Data source: Local CSV fallback"
    )


# =========================================================
# DATE RANGE
# =========================================================

min_date = df["Date"].min().date()

max_date = df["Date"].max().date()


date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


if (
    isinstance(date_range, tuple)
    and len(date_range) == 2
):

    start_date = pd.Timestamp(
        date_range[0]
    )

    end_date = pd.Timestamp(
        date_range[1]
    )

else:

    start_date = pd.Timestamp(
        min_date
    )

    end_date = pd.Timestamp(
        max_date
    )


filtered_df = df[
    (df["Date"] >= start_date)
    &
    (df["Date"] <= end_date)
].copy()


# =========================================================
# PAGE TITLE
# =========================================================

st.title("StockSense")

st.subheader(
    "Stock Market Analytics & ML Prediction Platform"
)

st.write(
    f"Currently analyzing: **{selected_stock_name}** "
    f"({selected_ticker})"
)


# =========================================================
# DATA SOURCE
# =========================================================

st.caption(
    f"Data source: {data_source} | "
    f"Historical period: {min_date} to {max_date}"
)


# =========================================================
# LATEST MARKET DATA
# =========================================================

latest = df.iloc[-1]


st.markdown("---")

st.markdown(
    "## Latest Market Data"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Close",
        f"INR {latest['Close']:.2f}"
    )


with col2:

    st.metric(
        "Open",
        f"INR {latest['Open']:.2f}"
    )


with col3:

    st.metric(
        "High",
        f"INR {latest['High']:.2f}"
    )


with col4:

    st.metric(
        "Low",
        f"INR {latest['Low']:.2f}"
    )


st.write(
    "Latest trading date: "
    f"{latest['Date'].strftime('%Y-%m-%d')}"
)


# =========================================================
# TRAIN SELECTED STOCK MODELS
# =========================================================

model_results = train_models(df)


if model_results is None:

    st.warning(
        "Not enough historical data to train "
        "the machine learning models."
    )

    st.stop()


logistic_model = (
    model_results["logistic_model"]
)

random_forest_model = (
    model_results["random_forest_model"]
)

metrics_df = model_results["metrics"]

prediction_history = (
    model_results["prediction_history"]
)


# =========================================================
# LATEST ML PREDICTION
# =========================================================

latest_ml_data = df.dropna(
    subset=ML_FEATURES
).copy()


latest_features = latest_ml_data[
    ML_FEATURES
].tail(1)


logistic_prediction = (
    logistic_model.predict(
        latest_features
    )[0]
)


random_forest_prediction = (
    random_forest_model.predict(
        latest_features
    )[0]
)


logistic_probability = (
    logistic_model.predict_proba(
        latest_features
    )[0]
)


random_forest_probability = (
    random_forest_model.predict_proba(
        latest_features
    )[0]
)


# =========================================================
# ML PREDICTION
# =========================================================

st.markdown("---")

st.subheader(
    "ML Next-Day Direction Prediction"
)


st.caption(
    "Model output is an educational UP/DOWN "
    "classification based on historical data. "
    "It is not a guaranteed prediction of future "
    "market movement or financial advice."
)


prediction_col1, prediction_col2 = (
    st.columns(2)
)


with prediction_col1:

    st.markdown(
        "### Logistic Regression"
    )

    if logistic_prediction == 1:

        st.success(
            "Prediction: UP"
        )

    else:

        st.error(
            "Prediction: DOWN"
        )


    st.write(
        "DOWN Probability: "
        f"{logistic_probability[0] * 100:.2f}%"
    )

    st.write(
        "UP Probability: "
        f"{logistic_probability[1] * 100:.2f}%"
    )


with prediction_col2:

    st.markdown(
        "### Random Forest"
    )

    if random_forest_prediction == 1:

        st.success(
            "Prediction: UP"
        )

    else:

        st.error(
            "Prediction: DOWN"
        )


    st.write(
        "DOWN Probability: "
        f"{random_forest_probability[0] * 100:.2f}%"
    )

    st.write(
        "UP Probability: "
        f"{random_forest_probability[1] * 100:.2f}%"
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("---")

st.subheader(
    "Model Performance"
)


st.dataframe(
    metrics_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1_Score": "{:.2%}"
    }),
    use_container_width=True
)


# =========================================================
# BACKTESTING SUMMARY
# =========================================================

st.markdown("---")

st.subheader(
    "Backtesting Summary"
)


total_predictions = len(
    prediction_history
)


logistic_correct = (
    prediction_history["Actual"]
    ==
    prediction_history[
        "Logistic_Prediction"
    ]
).sum()


random_forest_correct = (
    prediction_history["Actual"]
    ==
    prediction_history[
        "RandomForest_Prediction"
    ]
).sum()


logistic_accuracy = (
    logistic_correct
    / total_predictions
)


random_forest_accuracy = (
    random_forest_correct
    / total_predictions
)


backtest_col1, backtest_col2, backtest_col3 = (
    st.columns(3)
)


with backtest_col1:

    st.metric(
        "Test Predictions",
        total_predictions
    )


with backtest_col2:

    st.metric(
        "Logistic Accuracy",
        f"{logistic_accuracy:.2%}"
    )


with backtest_col3:

    st.metric(
        "Random Forest Accuracy",
        f"{random_forest_accuracy:.2%}"
    )


# =========================================================
# PREDICTION COMPARISON
# =========================================================

st.subheader(
    "Prediction Comparison"
)


comparison_df = prediction_history.copy()


comparison_df["Actual_Direction"] = (
    comparison_df["Actual"]
    .map({
        0: -1,
        1: 1
    })
)


comparison_df["Logistic_Direction"] = (
    comparison_df[
        "Logistic_Prediction"
    ].map({
        0: -1,
        1: 1
    })
)


comparison_df[
    "RandomForest_Direction"
] = (
    comparison_df[
        "RandomForest_Prediction"
    ].map({
        0: -1,
        1: 1
    })
)


fig_prediction = go.Figure()


fig_prediction.add_trace(
    go.Scatter(
        x=comparison_df["Date"],
        y=comparison_df[
            "Actual_Direction"
        ],
        mode="lines",
        name="Actual"
    )
)


fig_prediction.add_trace(
    go.Scatter(
        x=comparison_df["Date"],
        y=comparison_df[
            "Logistic_Direction"
        ],
        mode="lines",
        name="Logistic Regression"
    )
)


fig_prediction.add_trace(
    go.Scatter(
        x=comparison_df["Date"],
        y=comparison_df[
            "RandomForest_Direction"
        ],
        mode="lines",
        name="Random Forest"
    )
)


fig_prediction.update_layout(
    xaxis_title="Date",
    yaxis_title="Direction",
    height=450,
    yaxis=dict(
        tickmode="array",
        tickvals=[-1, 1],
        ticktext=["DOWN", "UP"]
    )
)


st.plotly_chart(
    fig_prediction,
    use_container_width=True
)


# =========================================================
# RECENT PREDICTION HISTORY
# =========================================================

st.subheader(
    "Recent Prediction History"
)


history_display = prediction_history[
    [
        "Date",
        "Close",
        "Actual_Label",
        "Logistic_Label",
        "RandomForest_Label"
    ]
].tail(20).sort_values(
    "Date",
    ascending=False
)


st.dataframe(
    history_display,
    use_container_width=True
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.markdown("---")

st.markdown(
    "## Dataset Overview"
)


col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Stock",
        selected_ticker
    )


with col2:

    st.metric(
        "Records",
        len(filtered_df)
    )


with col3:

    st.metric(
        "Start Date",
        filtered_df[
            "Date"
        ].min().strftime(
            "%Y-%m-%d"
        )
    )


with col4:

    st.metric(
        "End Date",
        filtered_df[
            "Date"
        ].max().strftime(
            "%Y-%m-%d"
        )
    )


# =========================================================
# CANDLESTICK CHART
# =========================================================

st.markdown("---")

st.subheader(
    "Candlestick Chart"
)


fig_candle = go.Figure()


fig_candle.add_trace(
    go.Candlestick(
        x=filtered_df["Date"],
        open=filtered_df["Open"],
        high=filtered_df["High"],
        low=filtered_df["Low"],
        close=filtered_df["Close"],
        name="Price"
    )
)


fig_candle.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    height=500,
    xaxis_rangeslider_visible=False
)


st.plotly_chart(
    fig_candle,
    use_container_width=True
)


# =========================================================
# PRICE + MOVING AVERAGES
# =========================================================

st.markdown("---")

st.subheader(
    "Price & Moving Averages"
)


fig_ma = go.Figure()


fig_ma.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Close"],
        mode="lines",
        name="Close"
    )
)


fig_ma.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["SMA_20"],
        mode="lines",
        name="SMA 20"
    )
)


fig_ma.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["SMA_50"],
        mode="lines",
        name="SMA 50"
    )
)


fig_ma.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["EMA_20"],
        mode="lines",
        name="EMA 20"
    )
)


fig_ma.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["EMA_50"],
        mode="lines",
        name="EMA 50"
    )
)


fig_ma.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    height=500
)


st.plotly_chart(
    fig_ma,
    use_container_width=True
)


# =========================================================
# RSI
# =========================================================

st.markdown("---")

st.subheader(
    "RSI (14)"
)


fig_rsi = go.Figure()


fig_rsi.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["RSI_14"],
        mode="lines",
        name="RSI 14"
    )
)


fig_rsi.add_hline(
    y=70,
    line_dash="dash",
    annotation_text="Overbought (70)"
)


fig_rsi.add_hline(
    y=30,
    line_dash="dash",
    annotation_text="Oversold (30)"
)


fig_rsi.update_layout(
    xaxis_title="Date",
    yaxis_title="RSI",
    height=400,
    yaxis=dict(
        range=[0, 100]
    )
)


st.plotly_chart(
    fig_rsi,
    use_container_width=True
)


# =========================================================
# VOLUME
# =========================================================

st.markdown("---")

st.subheader(
    "Trading Volume"
)


fig_volume = go.Figure()


fig_volume.add_trace(
    go.Bar(
        x=filtered_df["Date"],
        y=filtered_df["Volume"],
        name="Volume"
    )
)


fig_volume.update_layout(
    xaxis_title="Date",
    yaxis_title="Volume",
    height=400
)


st.plotly_chart(
    fig_volume,
    use_container_width=True
)


# =========================================================
# TECHNICAL INDICATORS TABLE
# =========================================================

st.markdown("---")

st.subheader(
    "Technical Indicators - Latest 20 Records"
)


indicator_columns = [
    "Date",
    "Close",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "EMA_50",
    "RSI_14"
]


indicator_df = filtered_df[
    indicator_columns
].tail(20).copy()


st.dataframe(
    indicator_df.style.format({
        "Close": "{:.2f}",
        "SMA_20": "{:.2f}",
        "SMA_50": "{:.2f}",
        "EMA_20": "{:.2f}",
        "EMA_50": "{:.2f}",
        "RSI_14": "{:.2f}"
    }),
    use_container_width=True
)


# =========================================================
# BASIC STATISTICS
# =========================================================

st.markdown("---")

st.subheader(
    "Basic Statistics"
)


stats_df = filtered_df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]
].describe().round(2)


st.dataframe(
    stats_df,
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "StockSense is an educational stock market analytics "
    "and machine learning project. Historical model results "
    "do not guarantee future performance and should not be "
    "treated as financial advice."
)