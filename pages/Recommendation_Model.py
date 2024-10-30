import streamlit as st
import pandas as pd
import pickle
from joblib import load
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, storage
import io

headers ={
    "authorization": st.secrets["Gemini"]["API_KEY"],
    "content-type": "application/json"
}

genai.configure(api_key=headers["authorization"])


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
        Recommdf = pd.read_csv(io.BytesIO(csv_data))
        return Recommdf

    # Streamlit UI
    filename = "Recommendation_DB.csv"  # Replace with your CSV file name in Firebase Storage
    
    try:
        Recommdf = load_csv_from_firebase(filename)
    except Exception as e:
        st.write("An error occurred while loading the file:", e)
        Recommdf = pd.DataFrame()  # Return empty DataFrame in case of error

    return Recommdf

# Function to load the model from Firebase Storage
def load_model_from_firebase(filename, model_type="pickle"):
    initialize_firebase()
    
    # Access the Firebase Storage bucket
    bucket = storage.bucket()
    blob = bucket.blob(filename)

    # Download the file as a byte stream
    model_data = blob.download_as_bytes()

    # Load the model depending on the type
    if model_type == "pickle":
        model = pickle.loads(model_data)
    elif model_type == "joblib":
        model = load(io.BytesIO(model_data))
    else:
        raise ValueError("Unsupported model type. Use 'pickle' or 'joblib'.")
    
    return model

# Store the data in Streamlit session state to avoid reloading
if "Recommdf" not in st.session_state:
    st.session_state.Recommdf = load_data()

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", generation_config=generation_config, safety_settings=safety_settings
)

convo = model.start_chat(history=[])

data = load_data()

label_encoders = {}
for column in [
    "DISTRICTNAME",
    "UNITNAME",
    "Accident_Spot",
    "Accident_Location",
    "Accident_SubLocation",
    "Main_Cause",
    "Severity",
    "Junction_Control",
    "Road_Character",
    "Road_Type",
    "Surface_Type",
    "Surface_Condition",
    "Road_Condition",
]:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Load the models from Firebase
loaded_model_pickle = load_model_from_firebase("random_forest_model.pkl", model_type="pickle")
loaded_model_joblib = load_model_from_firebase("random_forest_model.joblib", model_type="joblib")

# Prepare input data similar to X_test
input_data = {
    "DISTRICTNAME": [5],  # Default value, will be updated based on user selection
    # Provide example values for other columns as well
}

y = data[
    [
        "UNITNAME",
        "Accident_Spot",
        "Accident_Location",
        "Accident_SubLocation",
        "Main_Cause",
        "Severity",
        "Junction_Control",
        "Road_Character",
        "Road_Type",
        "Surface_Type",
        "Surface_Condition",
        "Road_Condition",
    ]
]

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data)

st.markdown(
    """
    <div style="background-color:#edeffc; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="text-align: center;">Road Infrastructure Analysis and Improvement Suggestions</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# Dropdown for selecting district name
st.subheader("Select District Name")
district_name = st.selectbox("",options=label_encoders["DISTRICTNAME"].classes_)

# Update input data with the selected district name
input_df["DISTRICTNAME"] = label_encoders["DISTRICTNAME"].transform([district_name])

# Predictions
predictions_pickle = loaded_model_pickle.predict(input_df)
predictions_joblib = loaded_model_joblib.predict(input_df)

# Convert predictions to human-readable format
prediction_textual = {}
for column in y.columns:
    encoder = label_encoders[column]
    prediction_textual[column] = encoder.inverse_transform([int(predictions_pickle[0][y.columns.get_loc(column)])])

# Convert predictions to a DataFrame for better presentation
prediction_df = pd.DataFrame(prediction_textual)

# prediction_df.to_csv('preds.csv', index=False)
# Presenting the prediction in a more presentable format
st.subheader("Conditions :")
st.write(prediction_df)

# prediction_df.to_csv('preds.csv', index=False)

# prediction  = pd.read_csv("preds.csv")

# Create a DataFrame from the description
description_text = """
- UNITNAME: Name of the specific police station or unit handling the accident report.
- Accident_Spot:  Description of the location where the accident occurred.
- Accident_Location: General location of the accident.
- Accident_SubLocation: Specific details about the accident location.
- Main_Cause: Primary cause attributed to the accident.
- Severity: Severity level of the accident.
- Junction_Control: Control measures at the accident junction.
- Road_Character: Characteristics of the road where the accident occurred.
- Road_Type: Type of road involved in the accident.
- Surface_Type: Type of road surface material.
- Surface_Condition: Condition of the road surface at the time of the accident.
- Road_Condition: Overall condition of the road contributing to the accident.
"""

# Concatenate the description DataFrame with your existing DataFrame
#final_input = f"This is my data for the district and other attributes of it:\n\n{description_text}\n\n{prediction_df}\n\nI want to give suggestion based on these attribute values and give some analysis so provide me analysis of each attribute and suggestion for same to improve the condition"

#final_input = f"This is my data for the district and other attributes of it:\n\n{description_text}\n\n{prediction}\n\nI want to give suggestions for improvement to reduce accidents ultimately based on these attribute values. Each attribute carries meaningful analytics, and suggestions for improvement should be specific, actionable, and supported by data analysis. The suggestions should address deficiencies observed in the data and provide steps on how conditions can be improved to reduce accidents."

final_input = f"This is my data for the district and other attributes of it:\n\n{description_text}\n\n{prediction_df}\n\nI want to give suggestions based on these attribute values and provide some analysis. Each attribute carries meaningful analytics, and the suggestions for improvement need to be specific, actionable, and supported by data analysis. The suggestions should address deficiencies observed in the data and provide steps on how conditions can be improved to reduce accidents. They should not be generic but rather valid, actionable, and with proof of their potential impact, I want the response to be organised as follows- First bullet point should talk about analytics of data / observation from data and the second point should give suggestions based on those analytics"

useitlater = ""  # Define the variable outside the if block

# Submit button
if final_input:
    # Send user message to conversation
    convo.send_message(final_input)
    # Display response
    st.subheader("Response:")
    
    response_text = convo.last.text.split('\n\n')
    
    for line in response_text:
        if line.startswith("**") and line.endswith("**"):
            # Extract the text between **
            heading_text = line.strip("**")
            st.markdown(
                f"""
                <div style="background-color:#edeffc; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                    <h2 style="color:#333333; font-size: 20px;">{heading_text}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            # For normal text, just display as is
            st.markdown(f"<span style='font-size: 15;'>{line}</span>", unsafe_allow_html=True)

    
    print(final_input)
    useitlater = response_text  # Assign the value to useitlater here

st.subheader("Enter your Query over here:")
user_input = st.text_input(" ")

# Text input field for user input
if st.button("Ask Query"):
    if final_input:
        Fuser_input=f"{useitlater}+ Based on this tell me in brief + {user_input}"
        # Send user message to conversation
        convo.send_message(Fuser_input)
        # Display response
        st.subheader("Response :")
        response_text = convo.last.text.split('\n\n')
        
        for line in response_text:
                # Extract the text between **
            heading_text = line.strip("**")
            st.markdown(f"**{heading_text}**")
