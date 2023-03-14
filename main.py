# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:07:33 2023

@author: serhiy
"""
import base64
from pathlib import Path
import datetime
#from utilities import load_bootstrap
import itertools
import streamlit as st
import pandas as pd 
import numpy as np
import pymongo

import plotly.express as px
import plotly.graph_objects as go

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from streamlit_extras.metric_cards import style_metric_cards
#from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.stateful_button import button as statefull_button
from streamlit_toggle import st_toggle_switch
from streamlit_extras.colored_header import colored_header
from prophet.plot import plot_components_plotly
from PIL import Image

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

favicon = Image.open("logo.ico")
#logo = Image.open("logo.png")
st.set_page_config(layout="wide",
                   page_title="Neptunus",
                   page_icon=favicon,
                   menu_items={
                    'Get Help': 'https://neptunus.com.ua',
                    'Report a bug': 'https://neptunus.com.ua',
                    'About': "Welcome to neptunus panel"
                })

padding_top = 0

st.markdown(f"""
    <style>
        .appview-container .main .block-container{{
        padding-top: {padding_top}rem;    }}
    </style>""",
    unsafe_allow_html=True,
)

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
        
#         html, body, [class*="css"]  {
# 		font-family: 'Montserrat', sans-serif;
# 		}
        
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    /* This code gets the first element on the sidebar,
    and overrides its default styling */
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

client = pymongo.MongoClient("mongodb+srv://serhiyandrejev:es8NDllb0OidQJRM@neptunus.zjfuwpj.mongodb.net/?retryWrites=true&w=majority")
db = client['neptunus']

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

def set_default_in_selectbox(options, default_value):
    options.remove(default_value)
    options = [default_value] + options
    return options

def filter_dataframe(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()
    
    with modification_container:
        try:
            to_filter_columns = st.multiselect("Filter dataframe on", 
                                               df.columns,
                                               default=config['filters']['to_filter_columns'])
        except:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        
        config_dict = {"to_filter_columns": None}
        
        config_dict['to_filter_columns'] = to_filter_columns
        for column in to_filter_columns:
            config_dict[column] = {}
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                
                try:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=config['filters'][column]['user_cat_input'])
                except:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()))
                    
                config_dict[column]['user_cat_input'] = user_cat_input
                df = df[df[column].isin(user_cat_input)]
                
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                
                try: 
                    user_num_input = right.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        config['filters'][column]['user_num_input'],
                        step=step, 
                    )
                except:
                    user_num_input = right.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step, 
                    )
                    
                config_dict[column]['user_num_input'] = user_num_input
                df = df[df[column].between(*user_num_input)]
                
            elif is_datetime64_any_dtype(df[column]):
                try:
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            config['filters']['user_date_input'][0],
                            config['filters']['user_date_input'][1],
                        ),
                    )
                except:
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    
                
                
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    config_dict[column]['user_date_input'] = user_date_input
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                try:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                        value = config['filters'][column]['user_text_input']
                    )
                except:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    
                config_dict[column]['user_text_input'] = user_text_input
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]
                    
    config['filters'] = config_dict                
    return df, config

#@st.cache_data
def group_raw_data(df, columns,
                   date_column,
                   target_column, 
                   config = None):
    
    #try:
        filtered_df, config = filter_dataframe(df, config=config)

        col1, col2 = st.columns(2)
        with col1:
            try: 
                groupby_type = st.selectbox("Select GroupBy type",
                                            options=set_default_in_selectbox(["sum", "count"],
                                                                             config['groupby_type']))
            except:
                groupby_type = st.selectbox("Select GroupBy type",
                                            options=["sum", "count"])
        with col2:
            try:
                period_type = st.selectbox("Select date period to groupby"
                                           , options=set_default_in_selectbox(['M', 'Y', 'D', 'W'],
                                                                              config['period_type']))
            except:
                period_type = st.selectbox("Select date period to groupby",
                                           options=['M', 'Y', 'D', 'W'])
            
        config['groupby_type'] = groupby_type
        config['period_type'] = period_type

        df = filtered_df.set_index(date_column).groupby(pd.Grouper(freq=period_type)).agg({target_column : groupby_type}).reset_index()
        st.dataframe(df, use_container_width=True)
        
    #except:
    #    st.warning("Something went wrong")
    
        return df, config
    
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

@st.cache_data
def run_forecast(seasonality_mode, add_holidays, holidays,
                 prophet_df, initial_days, horizon,
                 predict_period, predict_frequency):
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
    future = m.make_future_dataframe(periods=predict_period, freq = predict_frequency)
    forecast = m.predict(future)
    
    mape = tuning_results['mape'].min()
    
    return mape, forecast, m

@st.cache_data
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

#col1, col2= st.columns(2, gap="small")

st.markdown(" # Neptunus Forecasting Panel " + img_to_html('logo.png'), unsafe_allow_html=True)


#config = {'_user_id': "0101"}

if 'config' not in st.session_state:
    st.session_state['config'] = {}


reports = db['reports'].find({"_user_id": "0101"}, {"_id": 0})
reports = list(reports)
list_of_titles = []
for report in reports:
    list_of_titles.append(report['report_title'])
    
with st.sidebar:
    try:
        selected_report = st.selectbox("Load report",
                                       set_default_in_selectbox(["New report"] + list_of_titles,
                                                                st.session_state.config['report_title']),)
    except:
        selected_report = st.selectbox("Load report", ["New report"] + list_of_titles)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load", key = "Load"):
            for item in reports:
                if item['report_title'] == selected_report:
                    st.session_state.config = item
                    break
            else:
                st.session_state.config = {}
    
    with col2:
        if st.button("Save Report"):
            config = st.session_state.config.copy()
            config['timestamp'] = datetime.datetime.now()
            x = db['reports'].insert_one(config)
            reports = db['reports'].find({"_user_id": "0101"}, {"_id": 0})
            reports = list(reports)
            list_of_titles = []
            for report in reports:
                list_of_titles.append(report['report_title'])

col1, col2, col3 = st.columns([4,1,1])

#if 'report_title' not in st.session_state:
#    st.session_state['report_title'] = "Untitled Report"


try:
    colored_header(
    label=st.session_state.config['report_title'],
    color_name="blue-70",
    description="",
    )
except:
    colored_header(
    label="Untitled report",
    color_name="blue-70",
    description="",
    )

col1, col2, col3 = st.columns([2,4,8])          
with col1: 
    if st.checkbox("Edit title"):
        with col2:
            st.session_state.config['report_title'] = st.text_input("", label_visibility='collapsed', value = st.session_state.config['report_title'])

#st.session_state.config['report_title'] = st.session_state.config['report_title']

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
        try: date_column = st.selectbox("Select date column", options = set_default_in_selectbox(list(columns), st.session_state.config['date_column']))
        except: date_column = st.selectbox("Select date column", options = columns)
    with col2:
        try: target_column = st.selectbox("Select target column", options = set_default_in_selectbox(list(columns), st.session_state.config['target_column']))
        except: target_column = st.selectbox("Select target column", options = columns)
            
    st.session_state.config['date_column'] = date_column
    st.session_state.config['target_column'] = target_column
        
    try: need_to_group = st.checkbox("GroupBy Data", key = "need_to_group", value = st.session_state.config['need_to_group'])
    except: need_to_group = st.checkbox("GroupBy Data", key = "need_to_group")
    
    st.session_state.config['need_to_group'] = need_to_group
    
    if need_to_group:
        df, st.session_state.config = group_raw_data(df,
                                                     columns,
                                                     date_column,
                                                     target_column,
                                                     config = st.session_state.config)
    
    prophet_df = preprocessing_data(df, date_column, target_column)
    
    tab1, tab2 = st.tabs(["EDA", "Forecasting"], )
    
    with tab1:
        try:
            colored_header(
            label="Exploratory Data Analysis",
            description="Exploratory data analysis allows you to examine your data before training your model. ",
            color_name="blue-70",
            )
            
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
       # try:
            colored_header(
            label="Model Training",
            description="Specify model parameters, holidays(if necessary) and cross-validation parameters. You can find out more about cross-validation in our learning guide. ",
            color_name="blue-70",
            )
            
            try: add_holidays = st.checkbox("Add holidays", value=st.session_state.config['add_holidays'])
            except: add_holidays = st.checkbox("Add holidays")
            st.session_state.config['add_holidays'] = add_holidays
            
            if add_holidays:
                try: holidays = st.selectbox("Select country holidays", options = set_default_in_selectbox(['US', 'UA', 'UK'], st.session_state.config['holidays']))
                except: holidays = st.selectbox("Select country holidays", options = ['US', 'UA', 'UK'])
                st.session_state.config['holidays'] = holidays
            else: holidays = None
                
            try: seasonality_mode = st.selectbox("Select seasonality mode", options = set_default_in_selectbox(['additive', 'multiplicative'], st.session_state.config['seasonality_mode']))
            except: seasonality_mode = st.selectbox("Select seasonality mode", options =['additive', 'multiplicative'])
            st.session_state.config['seasonality_mode'] = seasonality_mode
            
            max_days = (pd.to_datetime(prophet_df['ds'].max()) - pd.to_datetime(prophet_df['ds'].min())).days
                
            try: initial_days = st.slider(label="Initial days",
                                         min_value=0, 
                                         max_value=round(0.90 * max_days),
                                         value = st.session_state.config['initial_days'])
            except: initial_days = st.slider(label="Initial days",
                                         min_value=0, 
                                         max_value=round(0.90 * max_days),
                                         value = round(0.75 * max_days))
            try:
                if st.session_state.config['period'] > round((max_days - initial_days) / 5):
                   st.session_state.config['period'] = round((max_days - initial_days) / 5)
                period = st.slider(label="Period of cross-validation",
                                   min_value = 0,
                                   max_value = round((max_days - initial_days) / 5),
                                   value = st.session_state.config['period'])
            except: period = st.slider(label="Period of cross-validation",
                                   min_value = 0,
                                   max_value = round((max_days - initial_days) / 5),
                                   value = round((max_days - initial_days) / 5))
            
            st.session_state.config['initial_days'] = initial_days
            st.session_state.config['period'] = period
            horizon = period
            
            try:
                predict_period = st.slider("Horizon after last date to predict",
                                            min_value=1,
                                            max_value=365,
                                            value=st.session_state.config['predict_period'])
            except:
                predict_period = st.slider("Horizon after last date to predict",
                                           min_value=1,
                                           max_value=365,
                                           value = 365)
                
            try:
                predict_frequency = st.selectbox("Select predict frequency",
                                                 options=set_default_in_selectbox(["D", "W", "2W", "M"], st.session_state.config['predict_frequency']))
            except:
                predict_frequency = st.selectbox("Select predict frequency",
                                                 options=["D", "W", "2W", "M"])
                
            st.session_state.config['predict_period'] = predict_period
            st.session_state.config['predict_frequency'] = predict_frequency
                
                
            if st.button("Run Forecast"):
                with st.spinner("Processing black magic"):
                    mape, forecast, m = run_forecast(seasonality_mode, add_holidays, holidays,
                                                     prophet_df, initial_days, horizon,
                                                     predict_period, predict_frequency)
                    
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
        #except:
        #    st.warning("Waiting for data")
        
#st.write(st.session_state.config)
#st.text(st.session_state.config)

