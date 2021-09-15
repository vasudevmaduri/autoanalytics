import streamlit as st
from PIL import Image

st.title("Auto Analytics")

image = Image.open("assets//logo.png")
st.image(image)

