import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import google.generativeai as genai
from pathlib import Path
import plotly.io as pio
import firebase_admin
from firebase_admin import credentials, storage
import io

headers ={
    "authorization": st.secrets["Gemini"]["API_KEY"],
    "content-type": "application/json"
}

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
        ICAdf = pd.read_csv(io.BytesIO(csv_data))
        return ICAdf

    # Streamlit UI
    filename = "ICA.csv"  # Replace with your CSV file name in Firebase Storage
    
    try:
        ICAdf = load_csv_from_firebase(filename)
    except Exception as e:
        st.write("An error occurred while loading the file:", e)
        ICAdf = pd.DataFrame()  # Return empty DataFrame in case of error

    return ICAdf

# Store the data in Streamlit session state to avoid reloading
if "ICAdf" not in st.session_state:
    st.session_state.ICAdf = load_data()

# Preprocess the data
def preprocess_data(data):
    categorical_cols = ['Accident_Classification', 'Accident_Spot', 'Accident_Location', 'Accident_SubLocation', 'Severity', 'Collision_Type', 'Junction_Control', 'Road_Character', 'Road_Type', 'Surface_Type', 'Surface_Condition', 'Road_Condition', 'Weather', 'Lane_Type', 'Side_Walk']
    numerical_cols = ['Year', 'Noofvehicle_involved']

    # Convert categorical columns to numeric
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes

    # One-hot encode categorical variables
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    categorical_pipeline = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_cols)], remainder='passthrough')
    data_encoded = categorical_pipeline.fit_transform(data)

    return data_encoded

# Train the model with optimized parameters
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reduce the number of estimators and max depth
    best_model = RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_split=10, min_samples_leaf=1, n_jobs=-1)
    best_model.fit(X_train, y_train)

    best_accuracy = best_model.score(X_test, y_test)

    return best_model, best_accuracy

# Visualize feature importances
def visualize_feature_importances(best_model, data):
    best_feature_importances = best_model.feature_importances_
    best_important_features = sorted(list(zip(data.columns, best_feature_importances)), key=lambda x: x[1], reverse=True)

    top_10_features = [feature for feature, _ in best_important_features[:10]]
    top_10_importances = [importance for _, importance in best_important_features[:10]]

    return best_important_features

def main():
    genai.configure(api_key=headers["authorization"])  # Replace 'Your_API_Key_Here' with your actual API key

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

    categorical_cols = ['Accident_Classification', 'Accident_Spot', 'Accident_Location', 'Accident_SubLocation', 'Severity', 'Collision_Type', 'Junction_Control', 'Road_Character', 'Road_Type', 'Surface_Type', 'Surface_Condition', 'Road_Condition', 'Weather', 'Lane_Type', 'Side_Walk']

    st.title('Factors contributing to multiple injuries and fatalities, and solutions.')

    # Read the CSV file into a DataFrame
    data = load_data()

    # Initialize original_labels dictionary outside the block
    original_labels = {}

    label_encoder = LabelEncoder()

    # Iterate through each column
    for col in data.columns:
        # Encode only if the column is non-numeric
        if data[col].dtype == 'object':
            data[col] = label_encoder.fit_transform(data[col])
            # Store the original labels
            original_labels[col] = label_encoder.classes_

    # Preprocess the data
    data_encoded = preprocess_data(data)
    X = data_encoded[:, :-1]
    y = data_encoded[:, -1].toarray().ravel()

    # Train the model
    best_model, best_accuracy = train_model(X, y)

    # Visualize feature importances
    feature_importances = visualize_feature_importances(best_model, data)

    # Create two columns layout
    left_column, right_column = st.columns(2)

    # Create a dropdown menu for selecting the column in the right column
    with right_column:
        selected_column = st.selectbox("Select an Attribute", categorical_cols)

        # Check if the selected column is numerical or categorical
        if selected_column in categorical_cols:  # Categorical
            # Display the frequency of each category in the selected column
            category_counts = data[selected_column].value_counts(normalize=True).reset_index()
            category_counts.columns = ['Category', 'Influence']

            # Decode labels if encoded
            if selected_column in original_labels:
                category_counts['Category'] = category_counts['Category'].replace(dict(enumerate(original_labels[selected_column])))

            # Plot the bar chart
            fig = px.bar(category_counts, x='Category', y='Influence', title=f'Influence of categories in {selected_column}', labels={'Influence': 'Influence', 'Category': 'Category'},width=870, height=510,template='none')

        else:  # Numerical
            # Display the histogram of the selected column
            st.write(f"Histogram of {selected_column}:")
            fig = px.histogram(data, x=selected_column, title=f'Histogram of {selected_column}')
        st.plotly_chart(fig)
        

    # Visualize feature importances in the left column
    with left_column:
        st.write('<span style="font-size:24px; font-weight:bold;">Most Contributing Attributes:</span>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(15, 10))  # Adjust the size as needed
        sns.barplot(x='Importance', y='Feature', data=pd.DataFrame(feature_importances, columns=['Feature', 'Importance']).head(10), ax=ax)
        ax.set_title('Top 10 Important Features')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        plt.savefig("dash.jpeg", format="jpeg", dpi=500)  # Adjust DPI as needed
        st.pyplot(fig)



    # Define image_path here
    image_path = Path("dash.jpeg")

    prompt = "The image showcases the dasboard, derived from the accident reports dataset of Karnataka, India. Assume the role of a Data Analyst and provide key observations and insights to aid police and traffic department in better decision making to ultimately reduce the accidents.Give detailed point explanation for each of the below mentioned points uniquely for Karnataka"

    image_part = {
        "mime_type" : "image/jpeg",
        "data" : image_path.read_bytes()
    }

    prompt_parts = [
        prompt , image_part
    ]

    st.title('Suggestions for preventing accidents')

    # Display a spinner while waiting for the response
    with st.spinner('Generating suggestions...'):
        response = model.generate_content(prompt_parts)
        suggestions = response.text

    # Once the response is acquired, display the suggestions
    st.write(suggestions)


if __name__ == "__main__":
    main()
