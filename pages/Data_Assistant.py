import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os
import base64
import firebase_admin
from firebase_admin import credentials, storage
import io

# Define API key
headers = {
    "authorization": st.secrets["PandasAI"]["API_KEY"],
    "content-type": "application/json"
}
os.environ["PANDASAI_API_KEY"] = headers["authorization"]

# Firebase initialization
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate({
            "type": st.secrets["firebase"]["type"],
            "project_id": st.secrets["firebase"]["project_id"],
            "private_key_id": st.secrets["firebase"]["private_key_id"],
            "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
            "client_email": st.secrets["firebase"]["client_email"],
            "client_id": st.secrets["firebase"]["client_id"],
            "auth_uri": st.secrets["firebase"]["auth_uri"],
            "token_uri": st.secrets["firebase"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
        })
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'datathon-dashboard-467d2.appspot.com'
        })

# Load data only once and store it in session state
@st.cache_resource
def load_data():
    initialize_firebase()

    def load_csv_from_firebase(filename):
        # Access the Firebase Storage bucket
        bucket = storage.bucket()
        blob = bucket.blob(filename)

        # Download the file as a byte stream and load into a DataFrame
        csv_data = blob.download_as_bytes()
        Chatdf = pd.read_csv(io.BytesIO(csv_data))
        return Chatdf

    filename = "Default_Accident_Data.csv"
    try:
        Chatdf = load_csv_from_firebase(filename)
    except Exception as e:
        st.write("An error occurred while loading the file:", e)
        Chatdf = pd.DataFrame()  # Return empty DataFrame in case of error

    return Chatdf

# Store the data in Streamlit session state to avoid reloading
if "Chatdf" not in st.session_state:
    st.session_state.Chatdf = load_data()

# Dictionary to store the extracted dataframes
data = {}

def main():
    st.title("Chat with Your Data ")

    # Initialize data as an empty dictionary
    data = {}

    # Side Menu Bar
    with st.sidebar:
        st.title("Configuration:⚙️")
        st.text("Data Setup: 📝")
        file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])
        st.markdown(":green[*Please ensure the first row has the column names.*]")

    # Load the default CSV file if no file is uploaded
    if file_upload is None:
        st.info(f"Upload your own file from sidebar for Customized Reports")
        # Load data from Firebase and add it to `data` dictionary
        data["Default Data"] = st.session_state.Chatdf
    else:
        data = extract_dataframes(file_upload)

    Chatdf = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
    st.dataframe(data[Chatdf])

    # Instantiate the BambooLLM
    llm = BambooLLM()

    # Instantiate the PandasAI agent
    analyst = get_agent(data, llm)

    # Start the chat with the PandasAI agent
    chat_window(analyst, data[Chatdf])


def chat_window(analyst, Chatdf):
    with st.chat_message("assistant"):
        st.text("Get instant answers to your runtime queries with Data Assistant.")

    # Initializing message history and chart path in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying the message history on re-run
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                st.write(message["response"])
            elif 'error' in message:
                st.text(message['error'])
            elif 'plot_data' in message:
                img = base64.b64decode(message['plot_data'])
                st.image(img)

    # Predefined questions for the user to click
    predefined_questions = [ 
        "Can you plot the number of accidents over the years?",
        "What are the top 5 districts suffering from Road Accidents?",
        "What are the top 3 Accident Sublocations for Road Accidents?",
        "What are the top 3 Collision Type causing Fatal Severity Road Accidents?"
    ]

    st.markdown("## Sample Questions:")

    # Display the predefined questions as buttons
    for question in predefined_questions:
        if st.button(question):
            process_question(analyst, question)

    # Explicit user queries
    user_question = st.chat_input("What are you curious about? ")

    if user_question:
        process_question(analyst, user_question)

    # Function to clear history
    def clear_chat_history():
        st.session_state.messages = []

    # Button to clear history
    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

def process_question(analyst, question):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "question": question})

    try:
        with st.spinner("Analyzing..."):
            response = analyst.chat(question)

            # Check if a plot has been generated and saved in the export directory
            plot_path = "exports/charts/temp_chart.png"
            if os.path.exists(plot_path):
                with open(plot_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                st.image(base64.b64decode(img_data))
                st.session_state.messages.append({"role": "assistant", "plot_data": img_data})
                os.remove(plot_path)
            else:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "response": response})

    except Exception as e:
        st.write(e)
        error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"
        st.session_state.messages.append({"role": "assistant", "error": error_message})

def get_agent(data, llm):
    """
    The function creates an agent on the dataframes extracted from the uploaded files
    Args: 
        data: A Dictionary with the dataframes extracted from the uploaded data
        llm: LLM object based on the ll type selected
    Output: PandasAI Agent
    """
    agent = Agent(list(data.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
    return agent

def extract_dataframes(file_buffer):
    """
    This function extracts dataframes from the given file buffer (uploaded file).
    Args: 
        file_buffer: file buffer (uploaded file)
    Processing: Based on the type of file read_csv or read_excel to extract the dataframes
    Output: 
        Chatdfs: A dictionary with the dataframes
    """
    Chatdfs = {}
    if file_buffer.name.endswith('.csv'):
        csv_name = file_buffer.name.split('.')[0]
        Chatdf = pd.read_csv(file_buffer)
        Chatdfs[csv_name] = Chatdf
    elif file_buffer.name.endswith(('.xlsx', '.xls')):
        xls = pd.ExcelFile(file_buffer)
        for sheet_name in xls.sheet_names:
            Chatdfs[sheet_name] = pd.read_excel(file_buffer, sheet_name=sheet_name)
    return Chatdfs

if __name__ == "__main__":
    main()
