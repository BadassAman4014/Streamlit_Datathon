import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage
import base64
import cv2
import numpy as np
import asyncio
from tensorflow import keras
from concurrent.futures import ThreadPoolExecutor
import time
from notificationapi_python_server_sdk import notificationapi
import csv
import tempfile
from keras.models import model_from_json
import numpy as np
import tensorflow as tf

class AccidentDetectionModel(object):

    class_nums = ['Accident', "No Accident"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_accident(self, img):
        self.preds = self.loaded_model.predict(img)
        return AccidentDetectionModel.class_nums[np.argmax(self.preds)], self.preds
    
# Define headers for the notification API
headers = {
    "clientId": st.secrets["CCTV"]["clientId"],
    "clientSecret": st.secrets["CCTV"]["clientSecret"],
    "email": st.secrets["CCTV"]["email"],
    "number": st.secrets["CCTV"]["number"],
    "content-type": "application/json"
}

# Set the Streamlit page to run in wide mode by default
st.set_page_config(layout="wide")

# Paths to the video files
video_paths = ["Videos/videoplayback.mp4", "Videos/videoplayback2.mp4", "Videos/videoplayback3.mp4"]

# Centered title
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    <h1 class="centered-title">Road Accident Detection</h1>
    """,
    unsafe_allow_html=True
)

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

# Fetch the model from Firebase Storage
def fetch_model_from_firebase():
    initialize_firebase()  # Ensure Firebase is initialized
    bucket = storage.bucket()
    blob = bucket.blob('CCTV.keras')  # Replace with the path to your model in Firebase Storage
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
        blob.download_to_filename(temp_model_file.name)
        model = keras.models.load_model(temp_model_file.name)  # Load model using Keras
    return model

# Initialize the model
model = fetch_model_from_firebase()
# model = ("CCTV_Models/CCTV.keras")
accident_detection_model = AccidentDetectionModel("CCTV_Models/CCTV.json", model)
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the last message time
last_message_time = 0
last_log_time = 0

# Set title for the log in the sidebar
st.sidebar.title("Accident Log")

# Create placeholder for the log in the sidebar
log_placeholder = st.sidebar.empty()

# Initialize the log list
log_list = []

# # CSV file path for logging
# csv_file_path = "accident_logs.csv"

# Function to write log entry to CSV and update Streamlit sidebar
def log_to_csv_and_sidebar(current_time, camera_id, probability, notification_sent):
    date = time.strftime('%Y-%m-%d', time.localtime(current_time))
    time_entry = time.strftime('%H:%M:%S', time.localtime(current_time))
    
    # with open(csv_file_path, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([date, time_entry, camera_id, probability, 'Yes' if notification_sent else 'No'])

    log_entry = (
        f"Date: {date},<br>"
        f"Time: {time_entry},<br>"
        f"Camera_ID : {camera_id},<br>"
        f"Probability: {probability}%,<br>"
        f"Notification Sent: {'Yes' if notification_sent else 'No'}"
    )
    log_list.append(log_entry)
    log_placeholder.markdown("<br><br>".join(log_list), unsafe_allow_html=True)

async def send_notification():
    current_time = time.time()
    notificationapi.init(
        headers["clientId"],
        headers["clientSecret"]
    )

    await notificationapi.send({
        "notificationId": "ksp_datathon",
        "user": {
            "id": headers["email"],
            "number": headers["number"]
        },
        "mergeTags": {
            "comment": f"Camera_ID : 16580 \nProbability of an accident at Kothanur, Bengaluru, Karnataka 560077.\n\nTime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}\n\nFor the exact location, click here: \nhttps://maps.app.goo.gl/YRGv6kR9SoTik5Sa7 ",
            "commentId": "testCommentId"
        }
    })

async def detect_accident(frame, camera_id):
    global last_message_time, last_log_time
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))

    pred, prob = accident_detection_model.predict_accident(roi[np.newaxis, :, :])
    if pred == "Accident":
        prob = round(prob[0][0] * 100, 2)
        cv2.rectangle(frame, (0, 0), (160, 20), (0, 0, 0), -1)
        cv2.putText(frame, pred + " " + str(prob), (10, 15), font, 0.5, (255, 255, 0), 2)
        current_time = time.time()
        notification_sent = False
        if prob > 99.60 and (time.time() - last_message_time) > 480:
            asyncio.create_task(send_notification())
            last_message_time = time.time()
            notification_sent = True
            
        if prob > 95.50 and (time.time() - last_log_time) > 50:
            log_to_csv_and_sidebar(current_time, camera_id, prob, notification_sent)
            last_log_time = time.time()
    
    return frame

async def stream_video(video_path, placeholder, width, height, camera_name, camera_id, detect=False):
    loop_count = 0
    while loop_count < 2:
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if detect:
                frame = await detect_accident(frame, camera_id)
            
            cv2.putText(frame, f"Camera_ID : {camera_id}", (20, frame.shape[0] - 20), font, 0.5, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            encoded_frame = base64.b64encode(frame_bytes).decode()
            video_str = f'''
                <h4>{camera_name}</h4>
                <img width="{width}" height="{height}" src="data:image/jpeg;base64,{encoded_frame}">
            '''
            placeholder.markdown(video_str, unsafe_allow_html=True)
            await asyncio.sleep(0.0003)
        loop_count += 1

async def main():
    col1, col2, col3 = st.columns(3)
    placeholders = [col1.empty(), col2.empty(), col3.empty()]
    await asyncio.gather(
        stream_video(video_paths[0], placeholders[0], 480, 320, "Camera 1", 1650, detect=True),
        stream_video(video_paths[1], placeholders[1], 480, 320, "Camera 2", 1990, detect=False),
        stream_video(video_paths[2], placeholders[2], 480, 320, "Camera 3", 89650, detect=False)
    )

executor = ThreadPoolExecutor()

if st.sidebar.button('Start'):
    asyncio.run(main())

executor.shutdown(wait=True)
