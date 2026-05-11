import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QSlider
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal


# 视频检测线程 (同时发送原始帧和检测帧)
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)          # 检测后的帧
    original_frame_signal = pyqtSignal(np.ndarray)         # 原始帧
    progress_signal = pyqtSignal(int, int)                 # (当前帧, 总帧数)
    finished_signal = pyqtSignal()

    def __init__(self, model, video_path):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self._is_paused = False
        self._is_stopped = False
        self.cap = None
        self.total_frames = 0

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.finished_signal.emit()
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while not self._is_stopped:
            if self._is_paused:
                self.msleep(50)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            # 发送原始帧
            self.original_frame_signal.emit(frame.copy())

            # 推理并绘图
            results = self.model(frame, device=0)
            annotated = results[0].plot()

            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.progress_signal.emit(current_frame, self.total_frames)
            self.change_pixmap_signal.emit(annotated)

        self.cap.release()
        self.finished_signal.emit()

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_stopped = True
        self._is_paused = False

    def seek_frame(self, frame_no):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交通场景识别系统 - 7类检测")
        self.setMinimumSize(1200, 700)

        self.model = YOLO("yolov8n.pt")

        self.current_image = None           # 原始图片
        self.annotated_image = None         # 检测结果图
        self.original_frame = None          # 当前原始视频帧

        self.video_thread = None
        self.is_video_paused = False

        self.recording = False
        self.video_writer = None
        self.slider_dragging = False

        # 状态栏
        self.status_bar = self.statusBar()
        assert self.status_bar is not None
        self.status_bar.showMessage("模型已加载：yolov8n.pt | 设备：GPU")

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # ---------- 显示区域（左右分栏） ----------
        display_layout = QHBoxLayout()

        # 左：原始画面
        left_layout = QVBoxLayout()
        self.original_label = QLabel("原始画面")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setStyleSheet("border: 2px dashed gray;")
        self.original_label.setMinimumSize(500, 400)
        left_layout.addWidget(self.original_label)

        # 右：检测结果
        self.result_label = QLabel("检测结果")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("border: 2px dashed gray;")
        self.result_label.setMinimumSize(500, 400)
        display_layout.addLayout(left_layout)
        display_layout.addWidget(self.result_label)
        main_layout.addLayout(display_layout)

        # ---------- 进度条（含播放按钮） ----------
        slider_layout = QHBoxLayout()
        self.btn_play_pause_slider = QPushButton("")       # 初始为空
        self.btn_play_pause_slider.setFixedWidth(40)
        self.btn_play_pause_slider.setEnabled(False)
        self.btn_play_pause_slider.clicked.connect(
            self.toggle_play_pause_from_slider)

        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(0)
        self.video_slider.sliderPressed.connect(self.on_slider_pressed)
        self.video_slider.sliderReleased.connect(self.on_slider_released)

        slider_layout.addWidget(self.btn_play_pause_slider)
        slider_layout.addWidget(self.video_slider)
        main_layout.addLayout(slider_layout)

        # ---------- 底部控制按钮区 ----------
        btn_layout = QHBoxLayout()

        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self.open_image)

        self.btn_batch = QPushButton("批量图片")
        self.btn_batch.clicked.connect(self.batch_detect)

        self.btn_open_video = QPushButton("打开视频")
        self.btn_open_video.clicked.connect(self.open_video)

        self.btn_pause_video = QPushButton("暂停")
        self.btn_pause_video.clicked.connect(self.pause_video)
        self.btn_pause_video.setEnabled(False)

        self.btn_stop_video = QPushButton("停止")
        self.btn_stop_video.clicked.connect(self.stop_video)
        self.btn_stop_video.setEnabled(False)

        self.btn_save = QPushButton("保存当前结果")
        self.btn_save.clicked.connect(self.save_current_result)

        self.chk_record = QPushButton("🔴 录制视频")
        self.chk_record.setCheckable(True)
        self.chk_record.clicked.connect(self.toggle_record)
        self.chk_record.setEnabled(False)                 # 只有视频打开时才可用

        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_batch)
        btn_layout.addWidget(self.btn_open_video)
        btn_layout.addWidget(self.btn_pause_video)
        btn_layout.addWidget(self.btn_stop_video)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.chk_record)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

    # ==================== 图片操作 ====================
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        self.current_image = cv2.imread(file_path)
        if self.current_image is None:
            QMessageBox.warning(self, "错误", "无法读取图片")
            return

        # 显示原始图片（左侧）
        self.display_image(self.current_image, self.original_label)

        # 推理并显示结果（右侧）
        results = self.model(self.current_image, device=0)
        self.annotated_image = results[0].plot()
        self.display_image(self.annotated_image, self.result_label)

        self.status_bar.showMessage(f"检测完成: {file_path}")

    def clear_display(self):
        self.original_label.clear()
        self.original_label.setText("原始画面")
        self.result_label.clear()
        self.result_label.setText("检测结果")
        self.current_image = None
        self.annotated_image = None

    def display_image(self, cv_img, label):
        """在指定 QLabel 上显示 OpenCV 图片"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h,
                          bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pix = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pix)

    def batch_detect(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        img_paths = []
        for ext in exts:
            img_paths.extend(Path(folder).glob(ext))
        if not img_paths:
            QMessageBox.information(self, "提示", "所选文件夹中没有支持的图片文件")
            return

        save_dir = Path(folder) / "detection_results"
        save_dir.mkdir(exist_ok=True)

        self.status_bar.showMessage(f"开始批量检测，共 {len(img_paths)} 张图片...")

        for idx, img_path in enumerate(img_paths, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            results = self.model(img, device=0)
            annotated = results[0].plot()
            out_path = save_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)

            if idx == 1:
                self.display_image(img, self.original_label)
                self.display_image(annotated, self.result_label)
            self.status_bar.showMessage(
                f"处理中: {idx}/{len(img_paths)} - {img_path.name}")

        self.status_bar.showMessage(f"批量检测完成，结果保存至: {save_dir}")
        QMessageBox.information(
            self, "完成", f"全部处理完毕，共 {len(img_paths)} 张\n结果保存在: {save_dir}")

    # ==================== 视频核心 ====================
    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not file_path:
            return

        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()

        self.video_thread = VideoThread(self.model, file_path)
        self.video_thread.original_frame_signal.connect(
            self.update_original_frame)
        self.video_thread.change_pixmap_signal.connect(
            self.update_result_frame)
        self.video_thread.progress_signal.connect(self.on_progress_updated)
        self.video_thread.finished_signal.connect(self.video_finished)

        self.btn_open_video.setEnabled(False)
        self.btn_pause_video.setEnabled(True)
        self.btn_stop_video.setEnabled(True)
        self.chk_record.setEnabled(True)                     # 可以开始录制

        self.video_slider.setEnabled(True)
        self.video_slider.setValue(0)
        self.slider_dragging = False

        # 初始化播放按钮状态
        self.btn_play_pause_slider.setEnabled(True)
        self.btn_play_pause_slider.setText("⏸")
        self.is_video_paused = False
        self.btn_pause_video.setText("暂停")

        self.video_thread.start()
        self.status_bar.showMessage(f"正在播放: {file_path}")

    def update_original_frame(self, cv_img):
        """显示原始帧（左侧）"""
        self.display_image(cv_img, self.original_label)

    def update_result_frame(self, cv_img):
        """显示检测结果（右侧），同时写入录制文件"""
        self.display_image(cv_img, self.result_label)
        if self.recording and self.video_writer is not None:
            self.video_writer.write(cv_img)

    def on_progress_updated(self, current_frame, total_frames):
        if not self.slider_dragging:
            self.video_slider.setMaximum(total_frames - 1)
            self.video_slider.setValue(current_frame)

    def on_slider_pressed(self):
        self.slider_dragging = True
        if self.video_thread is not None and not self.is_video_paused:
            self.video_thread.pause()

    def on_slider_released(self):
        self.slider_dragging = False
        if self.video_thread is not None:
            target_frame = self.video_slider.value()
            self.video_thread.seek_frame(target_frame)
            if not self.is_video_paused:
                self.video_thread.resume()

    def toggle_play_pause_from_slider(self):
        """进度条左侧播放按钮"""
        if self.video_thread is None or not self.video_thread.isRunning():
            return
        if self.is_video_paused:
            self.video_thread.resume()
            self.btn_play_pause_slider.setText("⏸")
            self.btn_pause_video.setText("暂停")
            self.status_bar.showMessage("继续播放")
            self.is_video_paused = False
        else:
            self.video_thread.pause()
            self.btn_play_pause_slider.setText("▶")
            self.btn_pause_video.setText("继续")
            self.status_bar.showMessage("已暂停")
            self.is_video_paused = True

    def pause_video(self):
        """底部暂停按钮"""
        if self.video_thread is None:
            return
        if self.is_video_paused:
            self.video_thread.resume()
            self.btn_pause_video.setText("暂停")
            self.btn_play_pause_slider.setText("⏸")
            self.status_bar.showMessage("继续播放")
            self.is_video_paused = False
        else:
            self.video_thread.pause()
            self.btn_pause_video.setText("继续")
            self.btn_play_pause_slider.setText("▶")
            self.status_bar.showMessage("已暂停")
            self.is_video_paused = True

    def stop_video(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait()
        self.video_finished()

    def video_finished(self):
        self.btn_open_video.setEnabled(True)
        self.btn_pause_video.setEnabled(False)
        self.btn_stop_video.setEnabled(False)
        self.chk_record.setEnabled(False)
        self.is_video_paused = False
        self.btn_pause_video.setText("暂停")
        self.status_bar.showMessage("视频播放结束或已停止")

        self.video_slider.setEnabled(False)
        self.btn_play_pause_slider.setEnabled(False)
        self.btn_play_pause_slider.setText("")              # 清空图标
        self.video_thread = None

        if self.recording:
            self.recording = False
            self.chk_record.setText("🔴 录制视频")
            self.chk_record.setChecked(False)
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

    # ==================== 保存与录制 ====================
    def save_current_result(self):
        # 保存右侧检测结果图片
        img_to_save = self.annotated_image if self.annotated_image is not None else self.current_image
        if img_to_save is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        default_name = "detection_result.jpg"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", default_name, "JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp)"
        )
        if file_path:
            cv2.imwrite(file_path, img_to_save)
            self.status_bar.showMessage(f"图片已保存: {file_path}")

    def toggle_record(self):
        if self.video_thread is None or not self.video_thread.isRunning():
            QMessageBox.warning(self, "警告", "请先打开视频")
            self.chk_record.setChecked(False)
            return

        if not self.recording:
            default_name = "output_video.avi"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存视频", default_name, "AVI (*.avi);;MP4 (*.mp4)"
            )
            if not file_path:
                self.chk_record.setChecked(False)
                return

            cap = self.video_thread.cap
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(
                *'XVID') if file_path.endswith('.avi') else cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                file_path, fourcc, fps, (width, height))
            self.recording = True
            self.chk_record.setText("⏹ 停止录制")
            self.status_bar.showMessage(f"正在录制: {file_path}")
        else:
            self.recording = False
            self.chk_record.setText("🔴 录制视频")
            self.chk_record.setChecked(False)
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.status_bar.showMessage("录制已停止，视频已保存")

    def closeEvent(self, event):
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
