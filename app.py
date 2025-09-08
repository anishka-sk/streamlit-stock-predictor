import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objs as go

# ------------------- Streamlit Page Config -------------------
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for dark theme ---
st.markdown("""
<style>
    .reportview-container {
        background: #111;
        color: #eee;
    }
    .stApp {
        background-color: #111;
        color: #eee;
    }
    .main .block-container {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .sidebar .sidebar-content {
        background-color: #222;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSlider label, .stSelectbox label {
        color: #ddd;
    }
</style>
""", unsafe_allow_html=True)

st.title("LSTM Model for Stock Price Prediction 📈")
st.markdown("This application predicts stock prices using a standalone LSTM deep learning model.")

# ------------------- Sidebar Settings -------------------
st.sidebar.title("Stock Settings")
stock_tickers = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "NVDA", "JPM", "V", "PG"]
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", stock_tickers)
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365 * 4))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
look_back = st.sidebar.slider("Lookback Window:", 10, 120, 60)
epochs = st.sidebar.slider("Epochs:", 10, 200, 20)
batch_size = st.sidebar.slider("Batch Size:", 16, 128, 32)
future_days = st.sidebar.slider("Future Days:", 1, 30, 10)

# ------------------- Main Execution -------------------
if st.sidebar.button("Run Prediction"):
    st.info(f"Running prediction for {selected_ticker} with a lookback period of {look_back} days.")

    @st.cache_data
    def get_data(ticker, start_date, end_date):
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    df = get_data(selected_ticker, start_date, end_date)

    if df is not None and not df.empty:
        df = df.reset_index()

        # ------------------- Use Only Close Price -------------------
        close_data = df[['Close']].values

        # Scale only close price
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        # Training data split
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:training_data_len]

        # Create dataset
        def create_dataset(dataset, look_back):
            X, y = [], []
            for i in range(look_back, len(dataset)):
                X.append(dataset[i - look_back:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train_data, look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # ------------------- Build & Train LSTM -------------------
        st.subheader("Building and Training the LSTM Model")
        lstm_model = Sequential()
        lstm_model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 1)))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(LSTM(100, return_sequences=False))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(Dense(50))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = lstm_model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0
        )

        # ------------------- Training Loss Plot -------------------
        st.subheader("Model Training History")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss'))
        fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
        fig_loss.update_layout(title="Training & Validation Loss",
                               xaxis_title="Epochs",
                               yaxis_title="Loss",
                               template="plotly_dark")
        st.plotly_chart(fig_loss, use_container_width=True)

        # ------------------- Future Predictions -------------------
        st.subheader("Next Predictions")

        future_predictions = []
        last_look_back_data = scaled_data[-look_back:]

        for _ in range(future_days):
            input_data = np.reshape(last_look_back_data, (1, look_back, 1))
            next_pred_scaled = lstm_model.predict(input_data, verbose=0)[0][0]
            next_pred = scaler.inverse_transform([[next_pred_scaled]])[0][0]
            future_predictions.append(next_pred)

            # Update look_back window with new prediction
            last_look_back_data = np.append(last_look_back_data[1:], [[next_pred_scaled]], axis=0)

        # Create prediction dataframe
        future_dates = [df['Date'].iloc[-1] + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_predictions})
        st.dataframe(prediction_df, use_container_width=True)

        # ------------------- Historical + Future Plot -------------------
        st.subheader("Historical and Future Price Prediction")
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                                        mode='lines', name='Historical Close'))
        fig_future.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted_Close'],
                                        mode='lines+markers', name='Predicted Future',
                                        line=dict(dash='dash', color='orange')))
        fig_future.update_layout(title=f"{selected_ticker} Historical and Future Prediction",
                                 xaxis_title="Date", yaxis_title="Stock Price",
                                 template="plotly_dark")
        st.plotly_chart(fig_future, use_container_width=True)

    else:
        st.error("No data found for the selected ticker. Please try a different one.")
