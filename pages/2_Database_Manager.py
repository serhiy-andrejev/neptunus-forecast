# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:59:58 2023

@author: serhiy
"""

import streamlit as st
from PIL import Image


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