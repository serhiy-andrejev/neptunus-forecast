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

logo = Image.open("logo.png")


st.title("Welcome to Neptunus")
st.markdown("""Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eros est, consectetur at eleifend non, semper vitae massa. In mollis, justo tempus auctor auctor, dolor erat mollis nulla, sed condimentum ex orci ac mauris. Aliquam fringilla blandit augue, vel tincidunt arcu dictum ut. Nullam volutpat vel felis vel sodales. In eu ultricies justo. Curabitur porttitor finibus nibh quis imperdiet. Maecenas maximus imperdiet nulla, mattis rhoncus tellus ultricies in. Proin sapien nulla, ultricies blandit ultrices non, aliquet sed purus.

Suspendisse ultricies condimentum feugiat. Cras fermentum lorem et enim ultricies posuere. Nunc sed fringilla libero, nec dapibus quam. Integer facilisis mi mi, non maximus lorem ultricies et. Praesent accumsan vel erat in laoreet. Pellentesque cursus justo et vulputate accumsan. Donec tincidunt dignissim auctor.

Nulla facilisi. Cras nisi metus, commodo dapibus nibh ut, feugiat scelerisque risus. Cras massa risus, efficitur sed erat vel, tempus mollis odio. Nulla id facilisis nisi. Nunc a felis nec felis pretium maximus. Donec eu fringilla diam. Sed ac urna quis mi accumsan egestas. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque viverra lacinia sapien, eu volutpat leo hendrerit nec. Suspendisse eget odio eros. In hac habitasse platea dictumst.

Integer ac ipsum imperdiet, interdum justo ut, dignissim magna. Morbi et mi blandit, vestibulum sem vel, lacinia elit. Nam ut semper tellus. Pellentesque ut sagittis metus. Morbi non porta mi. Vestibulum rhoncus tellus sed arcu faucibus fringilla. Proin a dui ac nunc suscipit cursus id in sapien. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nullam lacus est, euismod vitae augue bibendum, pharetra hendrerit sem. Cras posuere justo non rutrum malesuada. In hac habitasse platea dictumst.""")
with st.sidebar:
    st.image(logo)
    

