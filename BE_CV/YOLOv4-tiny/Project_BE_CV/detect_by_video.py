import numpy as np
import cv2
CONFIDENCE_THRESHOLD = 0.01
NMS_THRESHOLD = 0.5
net = cv2.dnn.readNet("yolov4-tiny-custom_best.weights", "yolov4-tiny-custom.cfg")#read network
class_label = ['mask_weared_incorrect', 'with_mask', 'without_mask']# label
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(640, 480), scale=1/255, swapRB=True,crop= False)
video_file = "input.mp4"
video = cv2.VideoCapture(video_file)

while(True):
    # Capture frame-by-frame
    ret, frame = video.read()
    frame = cv2.resize(frame, dsize=(640, 480)) # resize ảnh về đúng kích thức đầu vào
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD) # class là label, scores là độ tin cậy, boxes là các tọa đọ
    for idx, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
        if score > 0.5:
            print("Class: " + str(classid) + " Score:" + str(score))
            print("box", box)
            class_name = class_label[classid]
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (box[0], box[1] - 5)
            fontScale = 1
            if classid == 0:
                cv2.rectangle(frame, box, (255, 0, 0), 2)
                color = (255, 0, 0)
            elif classid == 1:
                cv2.rectangle(frame, box, (0, 255, 0), 2)
                color = (0, 255, 0)
            else:
                cv2.rectangle(frame, box, (0, 0, 255), 2)
                color = (0, 0, 255)
            thickness = 2
            frame = cv2.putText(frame, class_name, org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Display the resulting frame
    # show
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()