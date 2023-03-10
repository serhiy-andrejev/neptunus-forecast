# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:07:33 2023

@author: serhiy
"""
import base64
from pathlib import Path
#from utilities import load_bootstrap
import itertools
import streamlit as st
import pandas as pd 
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.stateful_button import button as statefull_button
from streamlit_extras.colored_header import colored_header
from prophet.plot import plot_components_plotly
from PIL import Image

favicon = Image.open("logo.ico")
#logo = Image.open("logo.png")
st.set_page_config(layout="wide",
                   page_title="Neptunus",
                   page_icon=favicon)

padding_top = 0

st.markdown(f"""
    <style>
        .appview-container .main .block-container{{
        padding-top: {padding_top}rem;    }}
    </style>""",
    unsafe_allow_html=True,
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        
        html, body, [class*="css"]  {
		font-family: 'Montserrat', sans-serif;
		}
        
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    /* This code gets the first element on the sidebar,
    and overrides its default styling */
    section[data-testid="stSidebar"] div:first-child {
        top: 0;
        height: 100vh;
    }
</style>
""",unsafe_allow_html=True)

hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''

st.markdown(hide_img_fs, unsafe_allow_html=True)

style_metric_cards(border_color = '#1f62b4',
                   background_color = "#f5f7fa",
                   border_left_color = '#1f62b4',
                   box_shadow = False)

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    columns = df.columns
    return df, columns

@st.cache_data
def preprocessing_data(df, date_column, target_column):
    #df[date_column] = pd.to_datetime(date_column)
    df = df.reset_index()[[date_column, target_column]].rename(columns = {target_column: 'y', date_column: 'ds'})
    return df

@st.cache_data
def line_plot(df):
    fig = px.line(df,
                  x = 'ds',
                  y = 'y',
                  )
    #fig.update_layout(template='solar', paper_bgcolor='#002b36')
    return fig

@st.cache_data
def box_plot(df):
    fig = px.box(df,
                  x = 'y',
                  orientation='h',
                  height=300
                  )
    #fig.update_layout(template='solar', paper_bgcolor='#002b36')
    return fig

#@st.cache_data
def group_raw_data(df, columns, date_column, target_column):
    try:
        filtered_df = dataframe_explorer(df)
        #groupby_column = st.selectbox("Select group column", options = columns)
        col1, col2 = st.columns(2)
        with col1:
            groupby_type = st.selectbox("Select GroupBy type", options=["sum", "count"])
        with col2:
            period_type = st.selectbox("Select date period to groupby", options=['M', 'Y', 'D', 'W'])
    
        df = filtered_df.set_index(date_column).groupby(pd.Grouper(freq=period_type)).agg({target_column : groupby_type}).reset_index()
        st.dataframe(df, use_container_width=True)
        #df = filtered_df.groupby(by=)
    except:
        st.warning("Something went wrong")
    return df

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

def forecast_plot(forecast, mape, prophet_df):
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    _fcst = forecast[['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper', 'yhat']].merge(prophet_df[1:-1], on = 'ds', how = 'outer')
    fig = go.Figure([
    go.Scatter(
        name='Historical data',
        x=_fcst['ds'],
        y=_fcst['y'],
        mode='lines',
        line=dict(color='rgb(31, 98, 180)'),
    ),
    go.Scatter(
        name='Model',
        x=_fcst['ds'],
        y=_fcst['yhat'],
        mode='lines',
        line=dict(color='rgb(165, 48, 35)'),
    ),
    go.Scatter(
        name='Model upper error',
        x=_fcst['ds'],
        y=_fcst['yhat_upper'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Model lower error',
        x=_fcst['ds'],
        y=_fcst['yhat_lower'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(214,39,40, 0.1)',
        fill='tonexty',
        showlegend=False
    ),
    go.Scatter(
        name='Trend',
        x=_fcst['ds'],
        y=_fcst['trend'],
        mode='lines',
        line=dict(color='rgb(255,127,14)'),
    ),
    go.Scatter(
        name='Trend upper error',
        x=_fcst['ds'],
        y=_fcst['trend_upper'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Trend lower error',
        x=_fcst['ds'],
        y=_fcst['trend_lower'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(255,127,14, 0.1)',
        fill='tonexty',
        showlegend=False
    )
    ])
    fig.update_layout(
        yaxis_title="Value",
        title='Mean Average Percentage Error: ' + str(mape) + '%',
        hovermode="x",
        height=800
    )
    return fig

col1, col2= st.columns(2, gap="small")

st.markdown(" # Neptunus Forecasting Panel " + img_to_html('logo.png'), unsafe_allow_html=True)

filepath = st.file_uploader("Choose a file")
if filepath is not None:
    df, columns = load_data(filepath)
    st.dataframe(df.head(), use_container_width=True)
    
    colored_header(
    label="Data Preparation",
    description="Set up your data. Select the columns with dates and target variable. If your data is in raw format, set up the grouping using the appropriate section.",
    color_name="blue-70",
    )
    col1, col2 = st.columns(2)
    with col1:
        date_column = st.selectbox("Select date column", options = columns)
    with col2:
        target_column = st.selectbox("Select target column", options = columns)
        
    need_to_group = st.checkbox("GroupBy Data", key = "need_to_group")
    if need_to_group:
        df = group_raw_data(df, columns, date_column, target_column)
    
    prophet_df = preprocessing_data(df, date_column, target_column)
    
    #st.dataframe(prophet_df)
    
   # try:
   #     st.dataframe(prophet_df.head(), use_container_width=True)
   # except: 
   #     st.warning("Select different columns")
    #df.index = pd.to_datetime(df[date_column])
    #df = df.drop(columns = date_column)
    
    tab1, tab2 = st.tabs(["EDA", "Forecasting"], )
    
    with tab1:
        try:
            colored_header(
            label="Exploratory Data Analysis",
            description="Exploratory data analysis allows you to examine your data before training your model. ",
            color_name="blue-70",
            )
            #st.write(prophet_df.rename(columns = {'ds': 'Date'})['Date'].describe(include = 'all'))
            #st.write(prophet_df.rename(columns = {'y': 'Target'})['Target'].describe(include = 'all'))
            
            first_date = pd.to_datetime(prophet_df['ds'].min()).date() 
            last_date = pd.to_datetime(prophet_df['ds'].max()).date() 
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric(label="Total rows", value=len(prophet_df))
            with col2: st.metric(label="First date", value=str(first_date))
            with col3: st.metric(label="Last date", value=str(last_date))
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric(label="Mean", value=round(prophet_df['y'].mean()))
            with col2: st.metric(label="STD", value=round(prophet_df['y'].std()))
            with col3: st.metric(label="Median", value=round(prophet_df['y'].median()))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric(label="Min", value=round(prophet_df['y'].min()))
            with col2: st.metric(label="25%", value=round(prophet_df['y'].quantile(0.25)))
            with col3: st.metric(label="75%", value=round(prophet_df['y'].quantile(0.75)))
            with col4: st.metric(label="Max", value=round(prophet_df['y'].max()))
            
            st.plotly_chart(line_plot(prophet_df),
                            use_container_width=True,
                            config = {"displaylogo": False}
                            )
            
            st.plotly_chart(box_plot(prophet_df),
                            use_container_width=True,
                            config = {"displaylogo": False}
                            )
        except:
            st.warning("Waiting for data")
    with tab2:
        try:
            colored_header(
            label="Model Training",
            description="Specify model parameters, holidays(if necessary) and cross-validation parameters. You can find out more about cross-validation in our learning guide. ",
            color_name="blue-70",
            )
            add_holidays = st.checkbox("Add holidays")
            if add_holidays:
                holidays = st.selectbox("Select country holidays", options = ['US', 'UA', 'UK'])
            seasonality_mode = st.selectbox("Select seasonality mode", options =['additive', 'multiplicative'])
            max_days = (pd.to_datetime(prophet_df['ds'].max()) - pd.to_datetime(prophet_df['ds'].min())).days
            
            initial_days = st.slider(label="Initial days",
                                     min_value=0, 
                                     max_value=round(0.90 * max_days),
                                     value = round(0.75 * max_days))
            period = st.slider(label="Period of cross-validation",
                               min_value = 0,
                               max_value = round((max_days - initial_days) / 5),
                               value = round((max_days - initial_days) / 5))
            horizon = period
            
            if st.button("Run Forecast"):
                with st.spinner("Processing black magic"):
                    param_grid = {'changepoint_prior_scale': [0.001, 0.05, 0.5, 1],
                                  'seasonality_prior_scale': [0.01, 1, 5],
                                  'seasonality_mode': [seasonality_mode]
                                  }
                    # Generate all combinations of parameters
                    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
                    # Create a list to store MAPE values for each combination
                    mapes = [] 
                    # Use cross validation to evaluate all parameters
                    for params in all_params:
                        # Fit a model using one parameter combination
                        m = Prophet(**params)
                        if add_holidays:
                            m.add_country_holidays(country_name=holidays)
                        m.fit(prophet_df)
                        # Cross-validation
                        df_cv = cross_validation(m, 
                                                 initial=str(initial_days) +' days',
                                                 period=str(initial_days) +' days',
                                                 horizon = str(horizon) + ' days',
                                                 parallel="processes")
                        # Model performance
                        df_p = performance_metrics(df_cv, rolling_window=1)
                        # Save model performance metrics
                        mapes.append(df_p['mape'].values[0])
                        
                    # Tuning results
                    tuning_results = pd.DataFrame(all_params)
                    tuning_results['mape'] = mapes
                    # Find the best parameters
                    best_params = all_params[np.argmin(mapes)]
                    
                    m = Prophet(**best_params).fit(prophet_df)  
                    future = m.make_future_dataframe(periods=75, freq = "W")
                    forecast = m.predict(future)
                    
                    mape = tuning_results['mape'].min()
                    colored_header(
                    label="Model plot",
                    description="By examining the chart below you will be able to assess the model predictions and compare them with your own historical data. ",
                    color_name="blue-70",
                    )
                    st.plotly_chart(forecast_plot(forecast, mape, prophet_df),
                                    use_container_width= True,
                                    config = {"displaylogo": False})
                    
                    colored_header(
                    label="Model components",
                    description="You can use these graphs to assess the components of the model. You can find out more about how the model works and its components in our tutorial.",
                    color_name="blue-70",
                    )
                    st.plotly_chart(plot_components_plotly(m, forecast),
                                    use_container_width= True,
                                    config = {"displaylogo": False})
            else:
                st.info("Select parameters and press button")
        except:
            st.warning("Waiting for data")
        