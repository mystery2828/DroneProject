import cv2
import time
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)
while cap.isOpened():
    ret,frame  = cap.read()
    image = frame
    cv2.imshow('Frame', frame)
    cv2.imwrite("data/record/akash/akash" + str(time.time()) + ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
