import numpy as np
import cv2
CONFIDENCE_THRESHOLD = 0.01
NMS_THRESHOLD = 0.5
net = cv2.dnn.readNet("yolov4-tiny-custom_best.weights", "yolov4-tiny-custom.cfg")#read network
class_label = ['mask_weared_incorrect', 'with_mask', 'without_mask']# label
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True,crop= False)
img_path = '../test_file/images/images (6).jpg'
frame = cv2.imread(img_path)
Width = frame.shape[1]
Height = frame.shape[0]
scale = 0.00392
frame = cv2.resize(frame, dsize=(416, 416)) # resize ảnh về đúng kích thức đầu vào
classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD) # class là label, scores là độ tin cậy, boxes là các tọa đọ
for idx, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
   if score >= 0.5:
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
        frame = cv2.putText(frame, str(score), org, font, fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow('frame',frame)
#cv2.imwrite("detect_images_file/images (6).jpg", frame)
# When everything done, release the capture
cv2.waitKey(0)
cv2.destroyAllWindows()