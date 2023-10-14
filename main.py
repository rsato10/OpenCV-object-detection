import cv2
import numpy as np
from gui_buttons import Buttons

#init buttons
button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 80)

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

# init camera
# device index or the name of a video file. 0 is first webcam 1 is second...
cap = cv2.VideoCapture(0)

#resolution change
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def click_button(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)      
        # polygon = np.array([[(530, 400), (600, 400), (600, 450), (530, 450)]])
        # is_inside_quit_btn = cv2.pointPolygonTest(polygon, (x, y), False) 
        # if is_inside_quit_btn > 0:
        #     exit()
            
frame_read = True

#create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while frame_read:

    def custom_button(name, polygon, color, text_loc):
        #create button. If we click in the button (camera window)  
        #use numpy array is faster than python and we are processing 30fps
        #cv2.rectangle(frame, (20, 230), (220, 130), (0, 0, 200), -1, ) 
        cv2.fillPoly(frame, polygon, (0, 0, 200)) 
        #txt needs to be bottom left cordinate of rectangle 
        cv2.putText(frame, name, text_loc, cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

    #ret = true if frames grabbed false if not
    #capture frame by frame
    ret, frame = cap.read()

    #get active buttons list    
    active_buttons = button.active_buttons_list()
    #print("Active buttons", active_buttons)

    #object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):

        #drawing bounding box around detected object
        (x, y, width, height) = bbox

        class_name = classes[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 0, 50), 3)

    button.display_buttons(frame)

    # quit_btn_polygon = np.array([[(530, 400), (600, 400), (600, 450), (530, 450)]])
    # quit_btn_color = (255, 255, 255)
    # quit_btn_loc = (532, 440)
    # custom_button("quit", quit_btn_polygon, quit_btn_color, quit_btn_loc)

    #diplay frame
    cv2.imshow("Frame", frame)

    #param is millisec between each frame grab if 0 it waits till key is pressed
    key = cv2.waitKey(1) 
    #27 is esc 
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

