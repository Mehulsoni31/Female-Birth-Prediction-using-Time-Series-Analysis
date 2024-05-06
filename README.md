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

```python
sns.relplot(x="DayOfWeek",y="births",hue="month_name",data=data,height=5,aspect=3)
```

![image](https://github.com/Mehulsoni31/Female-Birth-Prediction-using-Time-Series-Analysis/assets/71382200/71c21244-7048-45eb-a2bb-f2352af33052)

## Testing For Stationarity

```python
from statsmodels.tsa.stattools import adfuller
#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(births):
    result=adfuller(births)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
```

```python
adfuller_test(data['births'])
```

## Output
##### ADF Test Statistic : -4.808291253559765
####  p-value : 5.2434129901498554e-05
##### #Lags Used : 6
####  Number of Observations Used : 358
##### strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary

## Decomposing a time series into its trend, seasonal, and residual components.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
final = seasonal_decompose(data['births'],model='additive', period = 52) # annual=1,Quaterly=4,monthly=12,weekly=52
final.plot();
```

![image](https://github.com/Mehulsoni31/Female-Birth-Prediction-using-Time-Series-Analysis/assets/71382200/8caa2281-a1cf-4e3d-a964-ab038678b4ed)


## Plotting: 

### AutoCorrelation
![image](https://github.com/Mehulsoni31/Female-Birth-Prediction-using-Time-Series-Analysis/assets/71382200/954d14da-a817-4ec5-8230-aaca1785a014)

### Partial AutoCorrelation
![image](https://github.com/Mehulsoni31/Female-Birth-Prediction-using-Time-Series-Analysis/assets/71382200/a933a2da-eb1a-402a-aad1-d94bf41ccae7)


## ARIMA VS SARIMA

### Prediction Compsrison Between both Model

#### ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA

# Update the model creation line to use the new class
model = ARIMA(train['births'], order=(0, 0, 1))

# Keep the rest of the code as is
model = model.fit()
model.summary()
```

#### SARIMA

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
model1=SARIMAX(train['births'],order=(2,1,2),seasonal_order=(0,1,1,12))
results=model1.fit()
results.summary()
```

![image](https://github.com/Mehulsoni31/Female-Birth-Prediction-using-Time-Series-Analysis/assets/71382200/a8e02277-77b4-43a1-a115-c4a1e6f36a82)

![image](https://github.com/Mehulsoni31/Female-Birth-Prediction-using-Time-Series-Analysis/assets/71382200/59c425de-5f0a-463f-9d79-6e26462a5e3b)
