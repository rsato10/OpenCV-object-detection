import cv2

#Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")

#define object detection model
model = cv2.dnn_DetectionModel(net)

#320 because needs to be multiple of 32 and processed in square with deeplearning
#opencv pixels 1-255 while on dnn value 0-1
model.setInputParams(size=(320, 320), scale=1/255)

#load class lists
#in dnn_model/classes.txt we have all the classes we can detect ordered 0-etc
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

#init camera
# device index or the name of a video file. 0 is first webcam 1 is second...
cap = cv2.VideoCapture(0)

while True:
    #ret = true if frames grabbed false if not
    #capture frame by frame
    ret, frame = cap.read()

    #object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):

        #drawing bounding box around detected object
        (x, y, width, height) = bbox

        class_name = classes[class_id]

        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 0, 50), 3)

    print("class_ids: ", class_ids)
    print("scores: ", scores)
    print("bboxes: ", bboxes)

    #diplay frame
    cv2.imshow("Frame", frame)

    #param is millisec between each frame grab if 0 it waits till key is pressed
    cv2.waitKey(1) 


