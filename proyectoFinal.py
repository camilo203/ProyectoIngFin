import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.express as px
import statsmodels.api as sm
from plotly import graph_objs as go
from yahooquery import Screener
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

@st.cache_data
def get_crypto_tickers():
    s = Screener()
    data = s.get_screeners('all_cryptocurrencies_us', count=250)

    # data is in the quotes key
    dicts = data['all_cryptocurrencies_us']['quotes']

    symbols = [f"{d['shortName']} _ {d['symbol']}" for d in dicts]
    return symbols

#Funciones
def gbm_paths_reales(S0, mu, sigma, N, num_paths):
    #Se convierten las estadísticas a la temporalidad de la proyección.
    #Dado que se tiene retornos diarios, las estadísticas son diarias y, como la proyección también es diaria, no se requieren ajustes.
    mu_Ajustada = mu
    sigma_Ajustada = sigma

    paths = np.zeros((num_paths, N+1))
    paths[:, 0] = S0
    
    for i in range(num_paths):
        for j in range(1, N+1):

            #El modelo clásico de MBG tiene el promedio y la volatilidad ajustada a la temporalidad de la proyección.
            #En este caso recuerde que el parámetro (drift - 0.5*sigma**2) realmente converge a la media ajustada.
            paths[i, j] = paths[i, j-1] * np.exp(mu_Ajustada + sigma_Ajustada*np.random.normal())
    
    return paths

@st.cache_data
def df_to_windowed_df(dataframe, n=3):
    matrix = []
    for i in range(n+1, len(data)+1):
        if i == len(data):
            matrix.append([data.index[i-1]]+data["Adj Close"][i-4:i].values.tolist())
        else:
            matrix.append([data.index[i]]+data["Adj Close"][i-4:i].values.tolist())
    return pd.DataFrame(matrix, columns=["Date", "t-3", "t-2", "t-1", "t"])

@st.cache_data
def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

@st.cache_data
def get_data(ticker):
    return yf.download(ticker, start=(dt.datetime.today()-dt.timedelta(days=362*5)).strftime("%Y-%m-%d"))


symbols = get_crypto_tickers()
opt = st.selectbox('Select a crypto', symbols)
ticker = opt.split(" _ ")[1]

s_data = get_data(ticker)

st.title("Crypto Prices")
st.plotly_chart(px.line(s_data["Adj Close"], labels={"value": "Price", "index": "Date"}))




modelos = ["Geometric Brownian Motion","LSTM"]
modelo = st.selectbox("Select the model for the analysis:",modelos)

if modelo == modelos[0]:
    temporalidades = "Years", "Months", "Days"
    temporalidad = st.selectbox("Select the temporality for the analysis:",temporalidades)

    data = s_data.copy(deep=True)
    data["Date"] = data.index
    data.reset_index(drop=True, inplace=True)
    if temporalidad == "Years":
        data["Date"] = data["Date"].dt.year
        newDf = pd.DataFrame(data.groupby("Date")["Adj Close"].mean())
        newDf["LnReturns"] = np.log(newDf["Adj Close"]/newDf["Adj Close"].shift(1))
        newDf.LnReturns.iloc[0] = 0
    elif temporalidad == "Months":
        data.Date = data.Date.apply(lambda x: x.strftime("%Y-%m"))
        newDf = pd.DataFrame(data.groupby("Date")["Adj Close"].mean())
        newDf["LnReturns"] = np.log(newDf["Adj Close"]/newDf["Adj Close"].shift(1))
        newDf.LnReturns.iloc[0] = 0
    else:
        newDf = data.copy(deep=True)
        newDf["LnReturns"] = np.log(newDf["Adj Close"]/newDf["Adj Close"].shift(1))
        newDf.LnReturns.iloc[0] = 0

    newDf = newDf.copy(deep=True)
    st.write("Logarithmic returns",)
    st.plotly_chart(px.line(newDf["LnReturns"], labels={"value": "LnReturns", "index": "Date"}))

    mu = newDf.LnReturns.mean()
    sigma = newDf.LnReturns.std()
    S0 = s_data["Adj Close"].iloc[-1]
    num_paths = st.number_input("Number of paths", min_value=100, max_value=10000, value=1000, step=1)
    N = st.number_input("Number of periods", min_value=3, max_value=10000, value=5, step=1)
    GMBpaths = gbm_paths_reales(S0, mu, sigma, N, num_paths)
    fig = px.line(GMBpaths.T, labels={"value": "Price", "index": "Period"})
    fig.update_layout(showlegend=False,xaxis_range=[0,N+1])
    st.plotly_chart(fig)
    

    confidence = st.slider("Confidence interval", min_value=80, max_value=98, value=90, step=1)
    mini,maxi = (100-confidence)/2, 100-(100-confidence)/2

    forecast = GMBpaths[:,1:]
    means = forecast.mean(axis=0)
    lower = np.apply_along_axis(lambda x: np.percentile(x, mini), 0, forecast)
    upper = np.apply_along_axis(lambda x: np.percentile(x, maxi), 0, forecast)

    x = [x for x in range(1,N+1)]
    y = means.tolist()
    y_upper = upper.tolist()
    y_lower = lower.tolist()


    fig = go.Figure([
        go.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            showlegend=False
        ),
        go.Scatter(
            x=x+x[::-1], # x, then x reversed
            y=y_upper+y_lower[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    ])
    fig.update_layout(yaxis_range=[0,round(max(y_upper),2)+1])
    fig.update_yaxes(title_text="Price")
    fig.update_xaxes(title_text="Period")

    st.plotly_chart(fig)
    period = st.selectbox("Select the period for the analysis:",range(1,N+1))
    prices = GMBpaths[:,period]
    lower = np.percentile(prices, mini)
    upper = np.percentile(prices, maxi)
    fig = px.histogram(GMBpaths[:,period], labels={"value": "Price", "index": "Period"},histnorm='probability density')
    fig.update_layout(showlegend=False)
    fig.add_vline(x=GMBpaths[:,period].mean(), line_width=1, line_dash="dash", line_color="red", annotation_text="Mean: "+str(round(GMBpaths[:,period].mean(),2)))
    fig.add_vline(x=lower, line_width=1, line_dash="dash", line_color="green", annotation_text="Lower: "+str(round(lower,2)))
    fig.add_vline(x=upper, line_width=1, line_dash="dash", line_color="green", annotation_text="Upper: "+str(round(upper,2)))
    st.plotly_chart(fig)

elif modelo == modelos[1]:
    data = s_data.copy(deep=True)
    data = data[["Adj Close"]]
    windowed_df = df_to_windowed_df(data, 3)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    st.title("Train, validation and test sets")
    fig = go.Figure([
        go.Scatter(
            name='Train',
            x=dates_train,
            y=y_train,
            line=dict(color='blue'),
            mode='lines',
            showlegend=True
        ),
        go.Scatter(
            name='Validation',
            x=dates_val,
            y=y_val,
            line=dict(color='yellow'),
            mode='lines',
            showlegend=True),
        go.Scatter(
            name='Test',
            x=dates_test,
            y=y_test,
            line=dict(color='green'),
            mode='lines',
            showlegend=True)
    ])
    st.plotly_chart(fig)

    model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(50, activation='relu', return_sequences=True),
                    layers.Dropout(0.2),
                    layers.LSTM(100, activation='relu', return_sequences=False),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

    model.compile(loss='mse', 
                  optimizer=Adam(learning_rate=0.007),
                  metrics=['mean_absolute_error'])
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    train_predictions = model.predict(X_train).flatten()

    st.title("Training predictions")
    fig = go.Figure([
        go.Scatter(
            name='Training Predictions',
            x=dates_train,
            y=train_predictions,
            line=dict(color='blue'),
            mode='lines',
            showlegend=True
        ),
        go.Scatter(
            name='Training Observations',
            x=dates_train,
            y=y_train,
            line=dict(color='yellow'),
            mode='lines',
            showlegend=True)
    ])

    st.plotly_chart(fig)

    val_predictions = model.predict(X_val).flatten()
    st.title("Validation Predictions")
    fig = go.Figure([
        go.Scatter(
            name='Validation Predictions',
            x=dates_val,
            y=val_predictions,
            line=dict(color='blue'),
            mode='lines',
            showlegend=True
        ),
        go.Scatter(
            name='Validation Observations',
            x=dates_val,
            y=y_val,
            line=dict(color='yellow'),
            mode='lines',
            showlegend=True)
    ])

    st.plotly_chart(fig)
    st.title("Test Predictions")
    test_predictions = model.predict(X_test).flatten()
    fig = go.Figure([
        go.Scatter(
            name='Testing Predictions',
            x=dates_test,
            y=test_predictions,
            line=dict(color='blue'),
            mode='lines',
            showlegend=True
        ),
        go.Scatter(
            name='Testing Observations',
            x=dates_test,
            y=y_test,
            line=dict(color='yellow'),
            mode='lines',
            showlegend=True)
    ])

    st.plotly_chart(fig)


    nextd = windowed_df.to_numpy()[-1,2:].reshape((3, 1))

    nextd = nextd.astype(np.float32)

    proximo_precio = model.predict(np.array([nextd])).flatten()[0]

    st.title(f"The price of {ticker} in the next period is: {proximo_precio}")
    st.title(f"Across al testing period the acurracy was: {round((1-np.abs(test_predictions-y_test)/y_test).mean()*100,2)}%")