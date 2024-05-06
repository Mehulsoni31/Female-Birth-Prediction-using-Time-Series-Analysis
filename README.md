# Female-Birth-Prediction-using-Time-Series-Analysis
This project aims to develop a predictive model for forecasting the number of female births in the next 15 days using time series analysis techniques. The model is trained on historical birth data and leverages the power of time series analysis to capture the underlying patterns and trends in the data.

## Description

In many regions, predicting the number of births, especially female births, can be crucial for healthcare resource planning and policymaking. Time series analysis provides a powerful set of tools to analyze and forecast future values based on historical data patterns.
This project utilizes various time series analysis techniques, such as decomposition, trend analysis, seasonal adjustments, and autoregressive integrated moving average (ARIMA) models, to build an accurate predictive model for female births in the next 15 days.

## Features

### Data Preprocessing
The project includes steps for cleaning and preparing the birth data, handling missing values, and formatting the data for time series analysis.

### Exploratory Data Analysis (EDA)
EDA techniques are employed to gain insights into the birth data patterns, trends, and seasonality.
Time Series Decomposition: The time series is decomposed into its constituent components (trend, seasonality, and residuals) to better understand the underlying patterns.

### Model Training
Various time series forecasting models, such as ARIMA and exponential smoothing, are trained and evaluated on the historical birth data.
Model Selection: The best-performing model is selected based on evaluation metrics like mean squared error (MSE), root mean squared error (RMSE), and mean absolute percentage error (MAPE).

### Future Forecasting 
The selected model is used to forecast the number of female births for the next 15 days.

### Visualization 
The project includes visualizations of the historical birth data, decomposed time series components, and forecasted values for better understanding and interpretation


## Installation Library

```python
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")
import pandas as pd
import statsmodels.api as sm
import matplotlib
%matplotlib inline
import seaborn as sns
pd.options.display.max_rows=50
pd.options.display.max_columns=10


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
```

## Data Import

```python
import pandas as pd

data= pd.read_excel('/content/daily-total-female-births-CA.xlsx', engine='openpyxl')
```

## Data Presentation



index,date,births
0,1959-01-01 00:00:00,35
1,1959-01-02 00:00:00,32
2,1959-01-03 00:00:00,30
3,1959-01-04 00:00:00,31
4,1959-01-05 00:00:00,44
