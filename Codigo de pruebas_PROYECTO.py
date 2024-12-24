import cv2
import supervision as sv
from gtts import gTTS
import os
from ultralytics import YOLO

HOME = os.path.expanduser(r"C:/Users/Martin/Desktop/DETECCIÓN DE ENTORNO CON SISTEMA DE AUDIO/PROYECTO")

model = YOLO(r"C:\Users\Martin\Desktop\DETECCIÓN DE ENTORNO CON SISTEMA DE AUDIO\PROYECTO\runs_YOLO\content\runs\detect\train2\weights\best.pt")

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def speak_detections(detections):
    detected_labels = []
    for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        label = model.names[class_id]
        if label:
            detected_labels.append(label)
    
    if detected_labels:
        text_to_speak = "Se han detectado los siguientes objetos: " + ", ".join(detected_labels)
        print(text_to_speak)
        
        tts = gTTS(text_to_speak, lang='es')
        tts.save("detection_labels.mp3")
        os.system("start detection_labels.mp3")
    else:
        print("No se ha detectado nada.")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(source=frame, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imshow('Frame Anotado', annotated_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  
        break
    elif key == 13:
        if detections:
            speak_detections(detections)

cap.release()
cv2.destroyAllWindows()
