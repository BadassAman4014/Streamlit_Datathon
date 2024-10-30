import streamlit as st

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Page title
st.title("Power BI Analysis")

# Function to display Power BI iframe
def display_power_bi_report(url):
    iframe_code = f'''
    <iframe src="{url}" frameborder="0" allowFullScreen="true" style="width:100%; height:800px;"></iframe>
    '''
    st.markdown(iframe_code, unsafe_allow_html=True)

# List of reports with titles and URLs from secrets
reports = {
    "Temporal Analysis": st.secrets["powerbi"]["temporal_analysis"],
    "Roadtype Analysis": st.secrets["powerbi"]["roadtype_analysis"],
    "Landmark Vicinity Analysis": st.secrets["powerbi"]["landmark_vicinity_analysis"],
    "Road Signage Analysis": st.secrets["powerbi"]["road_signage_analysis"],
    "Pedestrian Analysis": st.secrets["powerbi"]["pedestrian_analysis"],
    "Accused and Victim Report Analysis": st.secrets["powerbi"]["accused_victim_report_analysis"],
}

# Create a row of buttons for each report
col_count = len(reports)
cols = st.columns(col_count)

# Store the URL of the selected report
selected_report = None
selected_title = None

# Add buttons to each column
for i, (title, url) in enumerate(reports.items()):
    with cols[i]:
        if st.button(f"**{title}**"):
            selected_report = url  # Store the URL of the selected report
            selected_title = title

# Display the selected report if any button was clicked
if selected_report:
    st.subheader("Selected Report: " + selected_title)
    display_power_bi_report(selected_report)
else:
    st.subheader("Click a button above to view a report.")
