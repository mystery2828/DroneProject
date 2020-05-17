from flask import Flask, render_template, Response
import cv2
import time

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
