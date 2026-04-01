import sys
import cv2
import numpy as np
import tensorflow as tf
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFileDialog, QWidget, QMenuBar, QStatusBar)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QThread, Signal

# --- Поток для обработки ---
class PredictionWorker(QThread):
    # Теперь передаем только кадр и отдельно данные нейросети
    image_processed = Signal(np.ndarray, str, float)

    def __init__(self):
        super().__init__()
        self.model = None
        self.running = False
        self.source = None

    def load_model(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input}
            )

    def set_source(self, path):
        self.source = path

    def run(self):
        if self.model is None:
            return

        self.running = True
        cap = cv2.VideoCapture(self.source)
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Подготовка кадра
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(rgb_frame, (180, 180))
            img_array = tf.keras.utils.img_to_array(input_img)
            img_tensor = tf.expand_dims(img_array, 0)

            # Предсказание
            prediction = self.model.predict(img_tensor, verbose=0)[0][0]
            # score = tf.nn.sigmoid(prediction[0][0]).numpy()
            
            if prediction < 0.5:
                label = "ОБНАРУЖЕН ПОЖАР"
                confidence = (1 - prediction) * 100
            else:
                label = "ВСЁ СПОКОЙНО"
                confidence = prediction * 100

            # Отправляем чистый кадр и текстовые данные отдельно
            self.image_processed.emit(frame, label, confidence)
            
            if isinstance(self.source, str) and self.source.endswith(('.mp4', '.avi', '.mov')):
                cv2.waitKey(1)
            else:
                break 

        cap.release()

# --- Основное окно ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fire Detector")
        self.resize(1000, 800)

        self.worker = PredictionWorker()
        self.worker.image_processed.connect(self.update_display)

        # 1. Создание Меню
        self.create_menu()

        # 2. Информационная панель (отдельное место для результата)
        self.info_label = QLabel("Модель не загружена. Выберите модель в меню.")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            QLabel {
                font-size: 18px; 
                font-weight: bold; 
                background-color: #333; 
                color: white; 
                padding: 10px;
                border-radius: 5px;
            }
        """)

        # Поле вывода изображения
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(1, 1)
        self.display_label.setStyleSheet("background-color: #111;")

        # Кнопки управления
        self.btn_img = QPushButton("Загрузить Фото")
        self.btn_vid = QPushButton("Загрузить Видео")
        self.btn_stop = QPushButton("Очистить")
        
        # Стили кнопок
        button_style = "QPushButton { height: 40px; font-size: 14px; }"
        self.btn_img.setStyleSheet(button_style)
        self.btn_vid.setStyleSheet(button_style)
        self.btn_stop.setStyleSheet(button_style)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.info_label) # Результат теперь сверху
        main_layout.addWidget(self.display_label, stretch=1)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_img)
        btn_layout.addWidget(self.btn_vid)
        btn_layout.addWidget(self.btn_stop)
        main_layout.addLayout(btn_layout)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Сигналы
        self.btn_img.clicked.connect(self.load_image)
        self.btn_vid.clicked.connect(self.load_video)
        self.btn_stop.clicked.connect(self.stop_worker)

    def create_menu(self):
        menubar = self.menuBar()
        model_menu = menubar.addMenu("Модели")

        load_model_act = QAction("Выбрать модель", self)
        load_model_act.triggered.connect(self.select_model_file)
        model_menu.addAction(load_model_act)

    def select_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель", "", "Keras Models (*.keras *.h5)")
        if path:
            self.info_label.setText("Загрузка модели... Пожалуйста, подождите.")
            QApplication.processEvents() # Чтобы текст обновился мгновенно
            self.worker.load_model(path)
            self.info_label.setText(f"Модель загружена: {path.split('/')[-1]}")
            self.info_label.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 10px;")

    def load_image(self):
        if not self.worker.model:
            self.info_label.setText("ОШИБКА: Сначала выберите модель в меню!")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть фото", "", "Images (*.png *.jpg *.jpeg)")
        if path: self.start_worker(path)

    def load_video(self):
        if not self.worker.model:
            self.info_label.setText("ОШИБКА: Сначала выберите модель в меню!")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть видео", "", "Videos (*.mp4 *.avi *.mov)")
        if path: self.start_worker(path)

    def start_worker(self, source):
        self.stop_worker()
        self.worker.set_source(source)
        self.worker.start()

    def stop_worker(self):
        # 1. Останавливаем поток
        self.worker.running = False
        self.worker.wait()
        
        # 2. Очищаем экран от последнего кадра
        self.display_label.clear() 
        
        # 3. Возвращаем стандартный текст и стиль
        self.info_label.setText("Обработка остановлена. Выберите новый файл.")
        self.info_label.setStyleSheet("""
            background-color: #333; 
            color: white; 
            padding: 10px; 
            font-weight: bold;
        """)

    def update_display(self, frame, label, confidence):
        # Проверка: если нажали стоп, игнорируем последние пришедшие кадры
        if not self.worker.running:
            return

        # Обновление текста и цвета статус-панели
        self.info_label.setText(f"{label} | Уверенность: {confidence:.2f}%")
        if "ПОЖАР" in label:
            self.info_label.setStyleSheet("background-color: #C62828; color: white; font-weight: bold; padding: 10px;")
        else:
            self.info_label.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 10px;")

        # Отрисовка кадра
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())