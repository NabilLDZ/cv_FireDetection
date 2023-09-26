#IMPORT LIBRARY
from flask import Flask, render_template, Response
import cv2
import supervision as sv
from ultralytics import YOLO
import requests

#INISIALISASI 
#App flask
app = Flask(__name__)
#Video Capture
#cap = cv2.VideoCapture(0)  #web camera
#cap = cv2.VideoCapture('rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp') #cctv camera
cap = cv2.VideoCapture('static/video/1.mp4') #Video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#Load model
model = YOLO("Model/best.pt")

# Bounding Box
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

#fungsi menghubungkan dengan telegram
def send_to_telegram(message):
    apiToken = '6109239191:AAHT5GciZk955h-His7nvKRL_XReRW8_9Vk'
    chatID = '-886133536'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)

#fungsi mengecek confidence objek dan mengirimkan pesan
def notifikasi_telegram(confide_skor):
    if confide_skor.size > 0:
        confide_skor = confide_skor[:1]
        confide_skor = float(confide_skor)
        if confide_skor >= 0.60:
            send_to_telegram("API Terdeteksi!!!!!")


#fungsi pendeteksian objek
def pendeteksian_objek():
    global frame
    ret, frame = cap.read()
    result = model(frame, conf=0.60, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]

    #menampilkan bounding box box dan label
    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections, 
        labels=labels
    ) 

    #fungsi cek confidence dan kirim Notifikasi telegram
    confide_skor = detections.confidence
    notifikasi_telegram(confide_skor)
        

#fungsi menampilkan frame 
def gen_frames():
    global frame  
    while True:
        #pendeteksian objek
        pendeteksian_objek()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' 
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


#route fungsi gen_frames
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


#route render file html (template)
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
