import keras.backend.tensorflow_backend as tb
from flask import Flask, render_template, Response

tb._SYMBOLIC_SCOPE.value = True
import operator
import cv2
from keras.models import model_from_json
from mtcnn import MTCNN

json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

name = {
    'akash': {'Name': 'Akash C',
              'Mobile': '9686178945',
              'Gender': 'Male',
              'Age': '21'},
    'ashwini': {'Name': 'Ashwini S V',
                'Mobile': '9008390524',
                'Gender': 'Female',
                'Age': '43'},
    'chethan': {'Name': 'Chethan D',
                'Mobile': '9449337486',
                'Gender': 'Male',
                'Age': '50'}
}


app = Flask(__name__)
cap = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while cap.isOpened():

        ret, frame = cap.read()
        detector = MTCNN()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        if results:

            for i in range(len(results)):

                bounding_box = results[i]['box']

                cv2.rectangle(image,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (255, 0, 0),
                              thickness=4)
                roi = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                      bounding_box[0]:bounding_box[0] + bounding_box[2]]

                # roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                roi = cv2.resize(roi, (64, 64))
                result = loaded_model.predict(roi.reshape(1, 64, 64, 3))
                prediction = {'akash': result[0][0],
                              'ashwin': result[0][1],
                              'pranav': result[0][2]
                              }
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                cv2.putText(image, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                (flag, encodedImage) = cv2.imencode('.jpg', image)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')
                # time.sleep(1)


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def return_anthony():
    while cap.isOpened():

        ret, frame = cap.read()
        detector = MTCNN()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        if results:

            for i in range(len(results)):

                bounding_box = results[i]['box']
                cv2.rectangle(image,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (255, 0, 0),
                              thickness=4)
                roi = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                      bounding_box[0]:bounding_box[0] + bounding_box[2]]

                # roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                roi = cv2.resize(roi, (64, 64))
                result = loaded_model.predict(roi.reshape(1, 64, 64, 3))
                prediction = {'akash': result[0][0],
                              'ashwin': result[0][1],
                              'pranav': result[0][2]
                              }

                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                return prediction[0][0]


@app.context_processor
def context_processor():
    return dict(return_anthony=return_anthony)


if __name__ == '__main__':
    app.run(debug=False, threaded=False)