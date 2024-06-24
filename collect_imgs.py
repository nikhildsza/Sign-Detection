import os

import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(0,number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR,str(j))):
        os.makedirs(os.path.join(DATA_DIR,str(j)))

    print(f'Collecting data for class {j}')

    while True:
        ret , frame = cap.read()
        cv2.putText(frame, text='Ready? Press "Q" ! :)', org=(100, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.3, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.imshow('Camera',frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter<dataset_size:
        ret,frame = cap.read()
        cv2.imshow('Camera',frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR,str(j),'{}.jpg'.format(counter)),frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()