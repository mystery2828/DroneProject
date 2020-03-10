import cv2
import time
from mtcnn import MTCNN

url1 = 'http://192.168.0.100:8080/video'
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
j = 0
while cap.isOpened():

    detector = MTCNN()
    ret, frame1 = cap.read()
    image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    j += 1
    ''' Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.'''
    if result:
        for i in range(len(result)):
            bounding_box = result[i]['box']

            cv2.rectangle(frame1,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0, 155, 255),
                          2)

            ''' Take the region of interest, that is the face from the frame '''
            roi = frame1[bounding_box[1]:bounding_box[1] + bounding_box[3],
                  bounding_box[0]:bounding_box[0] + bounding_box[2]]

    cv2.imshow("frame1", frame1)
    cv2.imshow("roi", roi)
    cv2.imwrite("data/train/akash/akash" + str(time.time()) + ".jpg", cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
