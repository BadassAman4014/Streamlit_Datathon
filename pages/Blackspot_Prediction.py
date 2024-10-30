import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
import streamlit as st
from streamlit_folium import folium_static as st_folium
import numpy as np
from scipy.spatial import ConvexHull
import google.generativeai as genai
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, storage
import io

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Load API key from secrets
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
        Blackdf = pd.read_csv(io.BytesIO(csv_data))
        return Blackdf

    # Streamlit UI
    filename = "datasetk.csv"  # Replace with your CSV file name in Firebase Storage
    
    try:
        Blackdf = load_csv_from_firebase(filename)
    except Exception as e:
        st.write("An error occurred while loading the file:", e)
        Blackdf = pd.DataFrame()  # Return empty DataFrame in case of error

    return Blackdf

# Store the data in Streamlit session state to avoid reloading
if "Blackdf" not in st.session_state:
    st.session_state.Blackdf = load_data()

def fetch_and_update_data(year_range, Collision_Type, Severity, month_range, District, Accident_Classification, num_markers, min_samples, show_blackspots, show_marker_cluster, show_heatmap):
    Blackdf = st.session_state.Blackdf  # Use the stored data
    update_heatmap(Blackdf, year_range, Collision_Type, Severity, month_range, District, Accident_Classification, num_markers, min_samples, show_blackspots, show_marker_cluster, show_heatmap)

def give_analysis():
    genai.configure(api_key=headers["authorization"])
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                  generation_config=generation_config)

    prompt = "Based on the image given about Blackspot analytics in the Karnataka India region. Assume the role of a data analyst. You have to present a police officer with the key observations and useful insights and analytics that will aid in decision-making to improve resource utilization, traffic management, junction control, traffic signal optimization, police bandobast, etc. Also, list all the hotspots (orange marker clusters) and predicted hotspots represented in black markers."
    image_path = Path("image.jpeg")

    # Check if the image exists
    if image_path.exists():
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_path.read_bytes()
        }

        prompt_parts = [
            prompt, image_part
        ]
        response = model.generate_content(prompt_parts)
        return response.text
    else:
        return "Image not found for analysis."

def update_heatmap(Blackdf, year_range, Collision_Type, Severity, month_range, District, Accident_Classification, num_markers, min_samples, show_blackspots, show_marker_cluster, show_heatmap):
    
    filtered_Blackdf = Blackdf[(Blackdf['Year_x'].between(year_range[0], year_range[1])) & 
                     (Blackdf['Collision_Type'].isin(Collision_Type)) & 
                     (Blackdf['Severity'].isin(Severity)) & 
                     (Blackdf['Month'].between(month_range[0], month_range[1])) & 
                     (Blackdf['DISTRICTNAME'] == District) &
                     (Blackdf['Accident_Classification'].isin(Accident_Classification))]

    if filtered_Blackdf.empty:
        st.warning("No data available for the selected filters.")
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=6)
    else:
        locations = filtered_Blackdf[['Latitude_x', 'Longitude_x']].values
        dbscan = DBSCAN(eps=0.01, min_samples=min_samples)
        filtered_Blackdf['cluster'] = dbscan.fit_predict(locations)
        cluster_counts = filtered_Blackdf.groupby('cluster').size().reset_index(name='count')
        dense_clusters = cluster_counts[cluster_counts['count'] >= num_markers]
        top_dense_clusters = dense_clusters.nlargest(num_markers, 'count')
        m = folium.Map(location=[filtered_Blackdf['Latitude_x'].mean(), filtered_Blackdf['Longitude_x'].mean()], zoom_start=10)

        if show_blackspots:
            for index, row in top_dense_clusters.iterrows():
                cluster_Blackdf = filtered_Blackdf[filtered_Blackdf['cluster'] == row['cluster']]
                points = cluster_Blackdf[['Latitude_x', 'Longitude_x']].values
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_centroid = np.mean(hull_points, axis=0)
                hull_lat, hull_lon = hull_centroid[0], hull_centroid[1]
                nearest_point_idx = np.argmin(np.sum((points - hull_centroid) ** 2, axis=1))
                nearest_point = points[nearest_point_idx]
                nearest_lat, nearest_lon = nearest_point[0], nearest_point[1]
                avg_collision_type = cluster_Blackdf.iloc[nearest_point_idx]['Collision_Type']
                avg_severity = cluster_Blackdf.iloc[nearest_point_idx]['Severity']
                popup_content = f"Density: {row['count']}<br>"
                popup_content += f"Collision Type: {avg_collision_type}<br>"
                popup_content += f"Severity: {avg_severity}<br>"
                folium.Marker(location=[nearest_lat, nearest_lon], popup=popup_content, icon=folium.Icon(color='black')).add_to(m)

        if show_heatmap:
            heat_map = HeatMap(locations)
            m.add_child(heat_map)

        if show_marker_cluster:
            marker_cluster = MarkerCluster().add_to(m)
            locations_markers = filtered_Blackdf[['Latitude_x', 'Longitude_x']].values.tolist()
            for loc in locations_markers:
                folium.Marker(loc).add_to(marker_cluster)

    st_folium(m, width=1410, height=750)

    with st.spinner("Getting analysis..."):
        texttt = give_analysis()
        card_style = """
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            background-color: #f7f7f7;
        """
        st.markdown(
            f'<div style="{card_style}">{texttt}</div>', 
            unsafe_allow_html=True
        )

st.markdown("<h1 style='text-align: center; color: black;'>Blackspot Analysis & Prediction</h1>", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: left; font-size: 35px;'>Slicers</h1>", unsafe_allow_html=True)

show_heatmap = st.sidebar.checkbox('Show Heatmap', value=False)
show_marker_cluster = st.sidebar.checkbox('Show Marker Cluster', value=True)
show_blackspots = st.sidebar.checkbox('Show Predicted Blackspots', value=True)

year_range = st.sidebar.slider(
    '**Select a range of years:**',
    int(st.session_state.Blackdf['Year_x'].min()), int(st.session_state.Blackdf['Year_x'].max()), (int(st.session_state.Blackdf['Year_x'].min()), int(st.session_state.Blackdf['Year_x'].max()))
)

month_range = st.sidebar.slider(
    '**Select a range of months:**',
    int(st.session_state.Blackdf['Month'].min()), int(st.session_state.Blackdf['Month'].max()), (int(st.session_state.Blackdf['Month'].min()), int(st.session_state.Blackdf['Month'].max()))
)

Accident_Classification_checkbox = st.sidebar.multiselect(
    'Accident Classification:',
    st.session_state.Blackdf['Accident_Classification'].unique()
)

Collision_Type_checkbox = st.sidebar.multiselect(
    'Collision Type:',
    st.session_state.Blackdf['Collision_Type'].unique()
)

Severity_checkbox = st.sidebar.multiselect(
    'Severity:',
    st.session_state.Blackdf['Severity'].unique()
)

District_checkbox = st.sidebar.selectbox(
    'District',
    st.session_state.Blackdf['DISTRICTNAME'].unique()
)

num_markers = st.sidebar.slider('Minimum number of markers for prediction:', 0, 100, 10)
min_samples = st.sidebar.slider('Minimum samples for clustering:', 1, 10, 5)

# Button to update the heatmap
if st.sidebar.button('Update Heatmap'):
    fetch_and_update_data(year_range, Collision_Type_checkbox, Severity_checkbox, month_range, District_checkbox, Accident_Classification_checkbox, num_markers, min_samples, show_blackspots, show_marker_cluster, show_heatmap)
