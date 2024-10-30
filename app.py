import streamlit as st
# Set page configuration to wide mode
st.set_page_config(layout="wide")
# Page title
st.title("Simple Streamlit Page")

# Display text
st.write("Welcome to my Streamlit app!")

# User input
user_name = st.text_input("Enter your name:")

# Display output based on input
if user_name:
    st.write(f"Hello, {user_name}! 👋")

# Slider example
age = st.slider("Select your age:", min_value=0, max_value=100, value=25)
st.write(f"Your age is {age}")

# Button example
if st.button("Say Hello"):
    st.write("Hello, Streamlit user!")

# Checkbox example
if st.checkbox("Show More"):
    st.write("Thank you for checking the box!")

# Selectbox example
option = st.selectbox("Choose a color:", ["Red", "Green", "Blue"])
st.write(f"You selected {option}")

# Sidebar example
st.sidebar.title("Sidebar")
sidebar_option = st.sidebar.radio("Choose an option:", ["Option 1", "Option 2", "Option 3"])
st.sidebar.write(f"Sidebar option selected: {sidebar_option}")
