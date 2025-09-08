
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set the page to a dark theme and wide layout
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for dark theme and styling ---
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

# --- Sidebar for user inputs (now with a selectbox for tickers) ---
st.sidebar.title("Stock Settings")
stock_tickers = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "NVDA", "JPM", "V", "PG"]
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", stock_tickers)
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365 * 2))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
look_back = st.sidebar.slider("Lookback Window:", 10, 120, 60)
epochs = st.sidebar.slider("Epochs:", 10, 200, 50)
batch_size = st.sidebar.slider("Batch Size:", 16, 128, 32)
future_days = st.sidebar.slider("Future Days:", 1, 30, 10)

if st.sidebar.button("Run Prediction"):
    st.info(f"Running prediction for {selected_ticker} with a lookback period of {look_back} days.")
    
    # --- Data Fetching and Preprocessing ---
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
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check if there is enough data for the lookback period
        if len(df) < look_back:
            st.error(f"Error: Not enough data for the lookback period of {look_back} days. Please select an earlier start date or a smaller lookback window.")
            st.stop()
        
        # Use only Close price for prediction
        data = df['Close'].values.reshape(-1, 1)

        # --- Display Stock Data in a Table ---
        st.subheader(f"Stock Data for {selected_ticker}")
        st.dataframe(df[['Date', 'Close']], use_container_width=True)
        
        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Splitting data
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:training_data_len, :]
        
        def create_dataset(dataset, look_back):
            X, y = [], []
            for i in range(look_back, len(dataset)):
                X.append(dataset[i-look_back:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train_data, look_back)
        
        # Check if X_train is empty after creation
        if len(X_train) == 0:
            st.error("Error: Not enough data to create training samples. Please adjust your 'Start Date' or 'Lookback Window'.")
            st.stop()

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # --- Building and Training the LSTM Model ---
        st.subheader("Building and Training the LSTM Model")
        with st.spinner("Training LSTM model..."):
            lstm_model = Sequential()
            lstm_model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 1)))
            lstm_model.add(Dropout(0.3))
            lstm_model.add(LSTM(100, return_sequences=False))
            lstm_model.add(Dropout(0.3))
            lstm_model.add(Dense(50))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            history = lstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[early_stopping], verbose=0)
            
        # --- Future Price Prediction ---
        st.subheader(f"Future Price Prediction for the Next {future_days} Days")
        
        future_predictions = []
        test_data = scaled_data[-look_back:]
        
        for _ in range(future_days):
            X_test = np.reshape(test_data, (1, look_back, 1))
            pred_price = lstm_model.predict(X_test, verbose=0)[0][0]
            future_predictions.append(pred_price)
            test_data = np.append(test_data[1:], pred_price).reshape(-1, 1)

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = scaler.inverse_transform(future_predictions).flatten()

        # Create a dataframe for future predictions
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": future_predictions
        })
        st.dataframe(prediction_df, use_container_width=True)

    else:
        st.error("No data found for the selected ticker. Please try a different one.")
