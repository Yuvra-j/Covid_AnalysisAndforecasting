
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load the COVID data and prepare it for modeling
    Note: This assumes you've already cleaned and processed the main dataset
    """
    covid_data = pd.read_csv("C:/Users/Yuvraj/Desktop/covid_analysis_report/data/processed/processed_covid_data.csv")
    covid_data['Date'] = pd.to_datetime(covid_data['Date'])
    return covid_data

def forecast_with_arima(country_data, forecast_days=30):
    """
    Forecast COVID cases using ARIMA model
    
    Parameters:
    country_data (DataFrame): DataFrame containing date and case data for a specific country
    forecast_days (int): Number of days to forecast
    
    Returns:
    DataFrame with forecasted values and confidence intervals
    """
    print(f"Forecasting with ARIMA for {forecast_days} days ahead...")
    
    ts_data = country_data.set_index('Date')['MA7_New_Confirmed'].asfreq('D')
    ts_data = ts_data.fillna(ts_data.bfill())
    
    model = ARIMA(ts_data, order=(1, 1, 1))
    model_fit = model.fit()
    
    forecast_result = model_fit.get_forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    
    forecast_values = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecast': forecast_values,
        'Lower_CI': conf_int.iloc[:, 0].values,
        'Upper_CI': conf_int.iloc[:, 1].values
    })
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(ts_data.index, ts_data, label='Historical Data')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='ARIMA Forecast', color='red')
    plt.fill_between(forecast_df['Date'], 
                    forecast_df['Lower_CI'], 
                    forecast_df['Upper_CI'], 
                    color='pink', alpha=0.3)
    plt.title(f'ARIMA Forecast of Daily New COVID-19 Cases')
    plt.xlabel('Date')
    plt.ylabel('New Cases (7-day MA)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('arima_forecast.png')
    
    return forecast_df

def forecast_with_prophet(country_data, forecast_days=30):
    """
    Forecast COVID cases using Facebook Prophet
    
    Parameters:
    country_data (DataFrame): DataFrame containing date and case data for a specific country
    forecast_days (int): Number of days to forecast
    
    Returns:
    DataFrame with forecasted values
    """
    print(f"Forecasting with Prophet for {forecast_days} days ahead...")
    
    try:
        
        prophet_data = country_data[['Date', 'MA7_New_Confirmed']].rename(
            columns={'Date': 'ds', 'MA7_New_Confirmed': 'y'})
        
        
        prophet_data['y'] = prophet_data['y'].clip(lower=0)
        
        
        model = Prophet(
            interval_width=0.95,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.01,  
            seasonality_prior_scale=0.1,   
            changepoint_range=0.8
        )
        
       
        model.fit(prophet_data)
        
        
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
       
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
        forecast_df = forecast_df.rename(columns={
            'ds': 'Date',
            'yhat': 'Forecast',
            'yhat_lower': 'Lower_CI',
            'yhat_upper': 'Upper_CI'
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"Error in Prophet forecasting: {str(e)}")
        # Return a DataFrame with NaN values if Prophet fails
        forecast_dates = pd.date_range(
            start=country_data['Date'].max() + pd.Timedelta(days=1),
            periods=forecast_days
        )
        return pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': np.nan,
            'Lower_CI': np.nan,
            'Upper_CI': np.nan
        })

def forecast_with_lstm(country_data, forecast_days=30, look_back=14):
    """
    Forecast COVID cases using LSTM (Long Short-Term Memory) neural network
    
    Parameters:
    country_data (DataFrame): DataFrame containing date and case data for a specific country
    forecast_days (int): Number of days to forecast
    look_back (int): Number of previous days to use as input features
    
    Returns:
    DataFrame with forecasted values
    """
    print(f"Forecasting with LSTM for {forecast_days} days ahead...")
    
    ts_data = country_data.set_index('Date')['MA7_New_Confirmed'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_data_scaled = scaler.fit_transform(ts_data)
    
    X, y = [], []
    for i in range(len(ts_data_scaled) - look_back):
        X.append(ts_data_scaled[i:i+look_back, 0])
        y.append(ts_data_scaled[i+look_back, 0])
    X, y = np.array(X), np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('lstm_training.png')

    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(np.reshape(y_test, (-1, 1)))
    y_pred_inv = scaler.inverse_transform(y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f'LSTM Test RMSE: {rmse:.2f}')
    future_predictions = []
    current_batch = ts_data_scaled[-look_back:].reshape((1, look_back, 1))
    
    for i in range(forecast_days):
        current_pred = model.predict(current_batch)[0]
        future_predictions.append(current_pred)
        
        current_batch = np.append(current_batch[:, 1:, :], 
                                 [[current_pred]], 
                                 axis=1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    last_date = country_data['Date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': future_predictions.flatten()
    })
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(country_data['Date'][-60:], country_data['MA7_New_Confirmed'][-60:], label='Historical Data')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='LSTM Forecast', color='red')
    plt.title(f'LSTM Forecast of Daily New COVID-19 Cases')
    plt.xlabel('Date')
    plt.ylabel('New Cases (7-day MA)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('lstm_forecast.png')
    
    return forecast_df

def compare_models(country_data, arima_forecast, prophet_forecast, lstm_forecast):
    """
    Compare the forecasts from different models
    
    Parameters:
    country_data (DataFrame): Original data
    arima_forecast, prophet_forecast, lstm_forecast (DataFrames): Model forecasts
    """
    print("Comparing forecasting models...")
    
    plt.figure(figsize=(14, 8))
    plt.plot(country_data['Date'][-60:], country_data['MA7_New_Confirmed'][-60:], 
             label='Historical Data', color='black', linewidth=2)
    plt.plot(arima_forecast['Date'], arima_forecast['Forecast'], 
             label='ARIMA Forecast', linestyle='--')
    plt.plot(prophet_forecast['Date'], prophet_forecast['Forecast'], 
             label='Prophet Forecast', linestyle='--')
    plt.plot(lstm_forecast['Date'], lstm_forecast['Forecast'], 
             label='LSTM Forecast', linestyle='--')
    plt.fill_between(arima_forecast['Date'], 
                    arima_forecast['Lower_CI'], 
                    arima_forecast['Upper_CI'], 
                    color='lightblue', alpha=0.3,
                    label='ARIMA 95% CI')
    plt.fill_between(prophet_forecast['Date'], 
                    prophet_forecast['Lower_CI'], 
                    prophet_forecast['Upper_CI'], 
                    color='lightgreen', alpha=0.3,
                    label='Prophet 95% CI')
    
    plt.title('Comparison of COVID-19 Forecasting Models')
    plt.xlabel('Date')
    plt.ylabel('New Cases (7-day MA)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    combined_forecast = pd.DataFrame({
        'Date': arima_forecast['Date'],
        'ARIMA_Forecast': arima_forecast['Forecast'],
        'Prophet_Forecast': prophet_forecast['Forecast'],
        'LSTM_Forecast': lstm_forecast['Forecast']
    })
    combined_forecast.to_csv('combined_forecasts.csv', index=False)
    
    return combined_forecast

def run_forecasting_models(country='US', forecast_days=30):
    """
    Run all forecasting models and compare results
    
    Parameters:
    country (str): Country to forecast
    forecast_days (int): Number of days to forecast
    """
    try:
        
        covid_data = load_and_prepare_data()
        country_data = covid_data[covid_data['Country/Region'] == country].copy()
        print(f"Running forecasting models for {country} with {len(country_data)} data points")

       
        forecast_results = {}
        
        
        forecast_dates = pd.date_range(
            start=country_data['Date'].max() + pd.Timedelta(days=1),
            periods=forecast_days
        )
        
        
        try:
            arima_forecast = forecast_with_arima(country_data, forecast_days)
            forecast_results['ARIMA_Forecast'] = arima_forecast['Forecast'].values
        except Exception as e:
            print(f"Error in ARIMA forecasting: {str(e)}")
            forecast_results['ARIMA_Forecast'] = np.full(forecast_days, np.nan)
        
        
        try:
            prophet_forecast = forecast_with_prophet(country_data, forecast_days)
            forecast_results['Prophet_Forecast'] = prophet_forecast['Forecast'].values
        except Exception as e:
            print(f"Error in Prophet forecasting: {str(e)}")
            forecast_results['Prophet_Forecast'] = np.full(forecast_days, np.nan)
        
        
        try:
            lstm_forecast = forecast_with_lstm(country_data, forecast_days)
            forecast_results['LSTM_Forecast'] = lstm_forecast['Forecast'].values
        except Exception as e:
            print(f"Error in LSTM forecasting: {str(e)}")
            forecast_results['LSTM_Forecast'] = np.full(forecast_days, np.nan)
        
        
        combined_forecast = pd.DataFrame({
            'Date': forecast_dates,
            **forecast_results
        })
        
        
        combined_forecast.to_csv('combined_forecasts.csv', index=False)
        
        print("Forecasting complete! Results saved to files.")
        return combined_forecast
        
    except Exception as e:
        print(f"Error in run_forecasting_models: {str(e)}")
    
        return pd.DataFrame({
            'Date': pd.date_range(
                start=datetime.now(),
                periods=forecast_days
            ),
            'ARIMA_Forecast': np.full(forecast_days, np.nan),
            'Prophet_Forecast': np.full(forecast_days, np.nan),
            'LSTM_Forecast': np.full(forecast_days, np.nan)
        })

if __name__ == "__main__":
    forecast_results = run_forecasting_models(country='US', forecast_days=30)
    print(forecast_results.head())
