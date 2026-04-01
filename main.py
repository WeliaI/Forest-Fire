import tensorflow as tf
import cv2
import numpy as np
MODEL_PATH = 'models/v3_62_12k.keras' 
model = tf.keras.models.load_model(MODEL_PATH)

VIDEO_PATH = 0
IMG_SIZE = (180, 180)
CONFIDENCE_THRESHOLD = 0.5

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX

print("Нажмите 'q', чтобы выйти...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, IMG_SIZE)
    
    img_array = tf.keras.utils.img_to_array(resized_frame)
    img_tensor = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_tensor, verbose=0)
    score = tf.nn.sigmoid(prediction[0][0]).numpy()

    if score < CONFIDENCE_THRESHOLD:
        label = "FIRE"
        color = (0, 0, 255)
        confidence = (1 - score) * 100
    else:
        label = "NORMAL"
        color = (0, 255, 0)
        confidence = score * 100

    text = f"{label}: {confidence:.2f}%"
    
    cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1) 
    cv2.putText(frame, text, (20, 45), font, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('Fire Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()