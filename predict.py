import operator
import cv2
from keras.models import model_from_json
from mtcnn import MTCNN

names = {'akash': {'Name: ': 'Akash C',
                   'USN: ': '1JS17EC008',
                   'Class: ': '6th sem ECE',
                   'Mob No.: ': '9686178945'}}

json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

categories = {0: 'akash', 1: 'ashwin', 2: 'pranav'}

while True:
    ret, frame = cap.read()
    detector = MTCNN()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
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

            # np.reshape(roi, (64,64,3))
            '''   roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # np.reshape(roi, (1, 64, 64, 1))
            '''
            result = loaded_model.predict(roi.reshape(1, 64, 64, 3))
            prediction = {'akash': result[0][0],
                          'ashwin': result[0][1],
                          'pranav': result[0][2]
                          }
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            cv2.putText(image, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
            # print(names[prediction[0][0]])
            sql = 'SELECT name Name,usn USN,class Class mob Mob_No from names WHERE name =' + str(prediction[0][0])
            cv2.imshow("Frame", image)
    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
