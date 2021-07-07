import cv2 
import numpy as np

# main setup

whT = 320

classes_file = 'coco.names'
class_names = []
conf_threshold = 0.5
nms_threshold = 0.3
with open(classes_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

model_cfg = 'yolov3.cfg'
model_weights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# main setup end

def find_objects(img):
    hT, wT, cT = img.shape
    bounding_box = []
    class_ids = []
    confidences = []
    detected_class_name = ""
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT) ,[0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_names = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_names)   
    net.getUnconnectedOutLayers()

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w/2), int((detection[1] * hT) - h/2)

                bounding_box.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bounding_box, confidences, conf_threshold, nms_threshold)

    for i in indices:
        
        i = i[0]
        box = bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        img = img[x : x + w, y : y + h]
        detected_class_name = class_names[class_ids[i]]
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        #cv2.putText(img, f'{class_names[class_ids[i]].upper()} {int(confidences[i] * 100)}%',
            #(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return img, detected_class_name.upper()
