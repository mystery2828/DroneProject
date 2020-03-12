import cv2
from mtcnn import MTCNN
from os import listdir

for file in listdir('D:\\DroneProject\\data\\record\\ashwin'):
    filename = 'D:\\DroneProject\\data\\record\\ashwin\\'+file
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
                          thickness= 4)
            roi = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                  bounding_box[0]:bounding_box[0] + bounding_box[2]]
            cv2.imwrite('D:\\DroneProject\\data\\train\\ashwin\\'+file, roi)

# cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

    # cv2.imwrite("resultimage.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# print(result)