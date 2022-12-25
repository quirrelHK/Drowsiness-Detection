from flask import Flask,render_template,Response
import cv2
import torch
import numpy as np

app=Flask(__name__)
camera=cv2.VideoCapture(0)
model = torch.hub.load(r'yolov5', 'custom', path='yolov5/runs/train/exp7/weights/best.pt', force_reload=True, source='local')
model.iou = 0.45
model.conf = 0.55

def generate_frames():
    global model
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            
            results = model(frame)
            # print(results)
            ret, buffer = cv2.imencode('.jpg', np.squeeze(results.render()))
            # ret, buffer1 = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # frame1 = buffer1.tobytes()
            # print(frame)
            # print(np.asarray(frame1).shape)
            
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
