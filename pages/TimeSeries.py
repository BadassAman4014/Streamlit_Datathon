import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from matplotlib import pyplot as plt
import google.generativeai as genai
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, storage
import io

# Headers for generative AI
headers = {
    "authorization": st.secrets["Gemini"]["API_KEY"],
    "content-type": "application/json"
}

# Initialize Firebase
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
            "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
            "universe_domain": st.secrets["firebase"]["universe_domain"]
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
        return pd.read_csv(io.BytesIO(csv_data))

    # Streamlit UI
    filename = "Accidentdata_TimeSeries.csv"  # Replace with your CSV file name in Firebase Storage
    
    try:
        return load_csv_from_firebase(filename)
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error

# Read the CSV file
df = load_data()

if not df.empty:
    # Convert 'ds' to datetime format
    df['ds'] = pd.to_datetime(df['Offense_Date'])

    # Aggregate the number of accidents per date ('ds') based on 'Accident_Classification'
    df_agg = df.groupby('ds')['Accident_Classification'].size().reset_index(name='Number_of_Accident')

    # Prepare data for Prophet
    df_prophet = pd.DataFrame({
        'ds': df_agg['ds'],
        'y': df_agg['Number_of_Accident'].astype(int)
    })

    # Fit the Prophet model
    m = Prophet(interval_width=0.90, daily_seasonality=True)
    model = m.fit(df_prophet)

    # Make future predictions
    future = m.make_future_dataframe(periods=365*5, freq='D')
    forecast = m.predict(future)

    # Plot components
    plt2 = m.plot_components(forecast)
    plt.savefig('forecast.jpeg', bbox_inches='tight')  # Save the plot with tight layout

    # Configure the generative AI
    genai.configure(api_key=headers["authorization"])

    # Set up the model for generative AI
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

    # Prepare the image for the prompt
    image_path = Path("forecast.jpeg")

    prompt = ("The image showcases time series forecasting for the number of accidents over time, "
              "derived from the accident reports dataset of Karnataka, India. Assume the role of a Data Analyst and "
              "provide key observations and insights in English to aid police and traffic department in better decision making "
              "to ultimately reduce the accidents. "
              "Give detailed point explanation for each of the below mentioned points uniquely for Karnataka."
              "1. Overall Trend Analysis: Identify significant trends in accident rates over the years. Highlight increases or decreases with COVID-19 impact."
              "2. Seasonal Pattern: Analyze how seasonal variations in the seasons in Karnataka, like monsoons, major festivals of Karnataka, impact accident rates."
              "3. Monthly Variations: Investigate recurring patterns in accident rates by month of the year and potential reasons."
              "4. Daily Accident Trends: Mention the specific days or time frames with elevated accident rates. Differentiate between weekdays and weekends to understand distinct accident patterns and the potential reasons behind them. Analyze the time and days of the week and potential reasons behind the pattern."
              "5. Additional Insights and Recommendations: Give in bullet points in detail. Provide targeted recommendations for resource allocation based on identified trends and peak periods. Suggest traffic police interventions, such as targeted patrols or awareness campaigns during high-risk periods. Propose strategies for junction control, traffic signal optimization, and overall traffic management to reduce accidents. Discuss deployment strategies for emergency services and coordination mechanisms during peak accident times.")

    image_part = {
        "mime_type": "image/jpeg",
        "data": image_path.read_bytes(),
    }

    prompt_parts = [prompt, image_part]

    # Generate AI insights
    response = model.generate_content(prompt_parts)

    # Streamlit App
    st.title('Time Series Forecasting')

    # Add text describing the overall trend graph
    st.markdown("The overall trend graph displays the average number of accidents that took place on days of the year.")

    # Display the interactive forecast plot
    st.pyplot(plt2)

    # Add a button to fetch AI insights
    if st.button('Get AI Insights'):
        st.subheader('AI Insights:')
        st.write(response.text)
else:
    st.warning("No data available to display.")
