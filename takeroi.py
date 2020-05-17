from os import listdir

import cv2
from mtcnn import MTCNN

for file in listdir('D:\\DroneProject\\Tensorflow_working\\data\\record\\akash'):
    filename = 'D:\\DroneProject\\Tensorflow_working\\data\\record\\akash\\' + file
    detector = MTCNN()
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    if result:
        for i in range(len(result)):
            bounding_box = result[i]['box']

            cv2.rectangle(image,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (255, 0, 0),
                          thickness=4)
            roi = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                  bounding_box[0]:bounding_box[0] + bounding_box[2]]
            cv2.imwrite('D:\\DroneProject\\Tensorflow_working\\data\\train\\akash\\' + file, roi)