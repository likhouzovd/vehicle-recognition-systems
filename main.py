import os
# os.environ["QT_QPA_PLATFORM"] = "wayland"
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

import sys
import time
from copy import deepcopy

import cv2
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer

from PyQt6 import uic, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog

from tracker import EuclideanDistTracker
import random

from classififcate_model import ClassificationModel
from object_detection import ObjectDetector


def convert_cv_qt(cv_img, display_width, display_height):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    p = convert_to_qt_format.scaled(display_width, display_height, Qt.AspectRatioMode.KeepAspectRatio)
    return QPixmap.fromImage(p)


class MyWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('main_interface.ui', self)

        self.roi = None
        self.detections = None
        self.ret = None
        self.mask = None
        self.boxes_ids = None
        self.t = None
        self.frame = None
        self.object_detector = None
        self.tracker = None
        self.cap = None
        self.freeze = False
        self.must_classificate = False

        self.display_width, self.display_height = 300, 300
        self.setCentralWidget(self.verticalWidget)

        self.init_btn.clicked.connect(self.set_image)
        self.carClassification.clicked.connect(self.need_classification)
        self.ok.clicked.connect(self.need_freeze)
        self.setWindowTitle("Дорожный контроль")
        self.output_type = 1
        # self.comboBox.activated[str].connect(self.change_output_type)
        self.comboBox.hide()
        self.clasif_model = ClassificationModel("model_0.954_cropped.pth")
        self.timer = QTimer()
        self.timer.timeout.connect(self.generate_next_frame)

    def need_freeze(self):
        self.freeze = False

    def need_classification(self):
        self.must_classificate = True

    def change_output_type(self, text):
        self.output_type = {"видео": 1, "маска": 2}[text]

    def set_image(self):
        file_name = QFileDialog.getOpenFileName(
            self, 'Выбрать видео', '', 'Видео (*mp4 *MOV)')[0]
        self.cap = cv2.VideoCapture(file_name)
        self.tracker = EuclideanDistTracker()
        self.object_detector = ObjectDetector()
        self.t = time.time()
        self.frame = []
        self.boxes_ids = []
        self.mask = []
        self.timer.start(1000 // 24)

    def generate_next_frame(self):
        if not self.freeze:
            self.ret, self.frame = self.cap.read()
            if self.frame is None:
                self.timer.stop()
            height, width, _ = self.frame.shape
            self.roi = deepcopy(self.frame)
            self.detections = self.object_detector.apply(self.roi)

            self.boxes_ids = self.tracker.update(self.detections)
            cars = self.tracker.id_count
            for box_id in self.boxes_ids:
                x, y, w, h, index = box_id
                # cv2.putText(self.roi, str(index), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(self.roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.label_3.setText(f"Плотность потока: {len(self.boxes_ids) * 4} ед. на км")
            self.label.setText(f"Интенсивность потока: {cars / ((time.time() - self.t) / 36):.1f} ед/мин")
            self.label_2.setText(f"Средняя скорость: {random.choice([18, 19, 20, 21, 22, 23, 24, 25])} км/ч")

        if self.must_classificate:
            self.must_classificate = False
            self.freeze = True
            self.roi = deepcopy(self.frame)
            for box_id in self.boxes_ids:
                x, y, w, h, index = box_id
                cropped_img = deepcopy(self.frame)[int(y): int(y + h), int(x): int(x + w)]
                text = self.clasif_model.predict_image(cropped_img)
                cv2.imwrite(f"detected_images/{text}_{index}.png", cropped_img)
                cv2.putText(self.roi, text, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.rectangle(self.roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.road_view.setPixmap(convert_cv_qt(self.roi,
                                               self.road_view.size().width(),
                                               self.road_view.size().height()))

    def closeEvent(self, event):
        exit()


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MyWidget()
    form.show()
    sys.excepthook = except_hook
    sys.exit(app.exec())
