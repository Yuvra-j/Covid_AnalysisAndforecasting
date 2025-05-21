COVID-19 Analysis and Forecasting (2023)

This is a project that focuses on analyzing and forecasting COVID-19 case data, specifically for the year 2023. It includes steps like data preprocessing, time-series analysis, visualizations, and using different models to predict future trends in COVID-19 cases.

Features:

- Analyzes confirmed, death, and recovered case counts
- Uses real-world data from 2023 only
- Forecasts future case trends using:
  - ARIMA model
  - Prophet model (by Meta)# not able to predict(forecast) properly so use arima or lstm
  - LSTM neural network
- Includes visualizations and plots of trends
- Performs cluster-based regional analysis

Project Structure:

- data/raw → original datasets (from Johns Hopkins University)
- data/processed → cleaned and merged data used for forecasting
- models → Python scripts for data processing and model building
- results → generated plots like forecasts and training curves
- scripts → main app or dashboard (optional)

Technologies Used:

- Python 3.12
- Pandas, Numpy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- Statsmodels (for ARIMA)
- Prophet
- TensorFlow / Keras (for LSTM)

Dataset Info:

The data used comes from the COVID-19 time-series dataset maintained by Johns Hopkins University. It includes both global and US-level data for confirmed cases, deaths, and recoveries. Only data from the year 2023 has been used in this project.

How to Run the Project:

1. First, clone the GitHub repository and move into the folder:

   git clone https://github.com/Yuvra-j/Covid_AnalysisAndforecasting.git  
   cd Covid_AnalysisAndforecasting

2. Run the data processing and model scripts:

   python models/data_processing.py  
   python models/forecasting_models.py

3. (Optional) If you have a dashboard set up, run the app with:

   streamlit run scripts/app.py

Sample Results:

The project generates prediction graphs using ARIMA and LSTM. These are saved in the results folder as PNG images.

Author:  
Yuvraj (GitHub: @Yuvra-j)
