import tensorflow as tf
import cv2
import numpy as np

# 1. Загрузка модели
# Если была ошибка с Lambda слоем, используем custom_objects
MODEL_PATH = 'forest_fire.keras' 
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Настройки
VIDEO_PATH = 'forest_fire.mp4'  # Укажите путь к вашему видео или 0 для веб-камеры
IMG_SIZE = (180, 180)           # Размер, на котором училась модель
CONFIDENCE_THRESHOLD = 0.5      # Порог уверенности

# Открываем видеопоток
cap = cv2.VideoCapture(VIDEO_PATH)

# Проверяем, открылось ли видео
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

# Настраиваем шрифт для текста
font = cv2.FONT_HERSHEY_SIMPLEX

print("Нажмите 'q', чтобы выйти...")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Видео закончилось

    # 3. Подготовка кадра для нейросети
    # OpenCV читает цвета как BGR, а Keras ждет RGB. Конвертируем:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Ресайз до 180x180 (как при обучении)
    resized_frame = cv2.resize(rgb_frame, IMG_SIZE)
    
    # Превращаем в массив и добавляем размерность батча (1, 180, 180, 3)
    img_array = tf.keras.utils.img_to_array(resized_frame)
    img_tensor = tf.expand_dims(img_array, 0)

    # 4. Предсказание
    # verbose=0 отключает лог для каждого кадра, чтобы не засорять консоль
    prediction = model.predict(img_tensor, verbose=0)
    score = tf.nn.sigmoid(prediction[0][0]).numpy()

    # Логика классов (0 - Пожар, 1 - Нет, если алфавитный порядок)
    # ВАЖНО: Проверьте свои классы. Обычно 0: Fire, 1: No_Fire
    if score < CONFIDENCE_THRESHOLD:
        label = "FIRE"
        color = (0, 0, 255) # Красный цвет (в BGR)
        confidence = (1 - score) * 100
    else:
        label = "NORMAL"
        color = (0, 255, 0) # Зеленый цвет
        confidence = score * 100

    # 5. Отрисовка результата на оригинальном кадре
    text = f"{label}: {confidence:.2f}%"
    
    # Рисуем плашку под текст для читаемости
    cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1) 
    cv2.putText(frame, text, (20, 45), font, 1, color, 2, cv2.LINE_AA)

    # Показываем кадр
    cv2.imshow('Fire Detection System', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()