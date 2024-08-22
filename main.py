import cv2
import numpy as np


def draw_label(img, text, pos, bg_color):
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
    text_w, text_h = text_size
    x, y = pos
    cv2.rectangle(img, (x, y), (x + text_w + 2, y + text_h + 4), bg_color, -1)
    cv2.putText(img, text, (x, y + text_h + 2), font, font_scale, (255, 255, 255), 1)


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)
            cv2.line(frame, (x, y), (x + 15, y), color, 3, lineType=cv2.LINE_AA)
            cv2.line(frame, (x, y), (x, y + 15), color, 3, lineType=cv2.LINE_AA)
            cv2.line(frame, (x + w, y), (x + w - 15, y), color, 3, lineType=cv2.LINE_AA)
            cv2.line(frame, (x + w, y), (x + w, y + 15), color, 3, lineType=cv2.LINE_AA)
            cv2.line(frame, (x, y + h), (x + 15, y + h), color, 3, lineType=cv2.LINE_AA)
            cv2.line(frame, (x, y + h), (x, y + h - 15), color, 3, lineType=cv2.LINE_AA)
            cv2.line(frame, (x + w, y + h), (x + w - 15, y + h), color, 3, lineType=cv2.LINE_AA)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - 15), color, 3, lineType=cv2.LINE_AA)

            draw_label(frame, f'{label}: {int(confidence * 100)}%', (x, y - 20), color)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
