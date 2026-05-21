"""PySide6/PyQtGraph interactive viewer for PtyRAD objp data."""

from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from config import COLORMAP_OPTIONS, DISPLAY_MODE_OPTIONS, PROCESSING_STATE
from data_processor import DataProcessor, ParameterManager
from file_utils import check_if_processed
from video_generator import VideoGenerator


def _create_app_icon():
    """Create a distinctive runtime icon for the Qt window/taskbar."""
    icon = QtGui.QIcon()
    for size in (32, 48, 64, 128, 256):
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        scale = size / 256.0
        rect = QtCore.QRectF(10 * scale, 10 * scale, 236 * scale, 236 * scale)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#050b12"))
        painter.drawRoundedRect(rect, 46 * scale, 46 * scale)

        painter.setPen(QtGui.QPen(QtGui.QColor("#1d4ed8"), 8 * scale,
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawRoundedRect(QtCore.QRectF(28 * scale, 28 * scale, 200 * scale, 200 * scale),
                                34 * scale, 34 * scale)

        painter.setPen(QtGui.QPen(QtGui.QColor("#60a5fa"), 11 * scale,
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                  QtCore.Qt.RoundJoin))
        wave = QtGui.QPainterPath()
        wave.moveTo(46 * scale, 154 * scale)
        wave.cubicTo(74 * scale, 80 * scale, 110 * scale, 210 * scale, 138 * scale, 128 * scale)
        wave.cubicTo(160 * scale, 66 * scale, 196 * scale, 102 * scale, 216 * scale, 70 * scale)
        painter.drawPath(wave)

        painter.setPen(QtGui.QPen(QtGui.QColor("#eaf3ff"), 8 * scale,
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawLine(QtCore.QPointF(70 * scale, 196 * scale), QtCore.QPointF(188 * scale, 196 * scale))
        painter.drawLine(QtCore.QPointF(88 * scale, 54 * scale), QtCore.QPointF(88 * scale, 196 * scale))

        painter.setPen(QtCore.Qt.NoPen)
        for x, y in ((88, 54), (70, 196), (188, 196), (216, 70), (138, 128)):
            painter.setBrush(QtGui.QColor("#f8fbff"))
            painter.drawEllipse(QtCore.QPointF(x * scale, y * scale), 8 * scale, 8 * scale)
            painter.setBrush(QtGui.QColor("#2563eb"))
            painter.drawEllipse(QtCore.QPointF(x * scale, y * scale), 4 * scale, 4 * scale)
        painter.end()

        icon.addPixmap(pixmap)
    return icon


def _set_windows_app_id():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "dongzehao.plot_ptyrad.qt"
        )
    except Exception:
        pass


class LRUCache:
    """Small bounded cache for preview arrays."""

    def __init__(self, max_items=32):
        self.max_items = max_items
        self._items = OrderedDict()

    def get(self, key):
        if key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def set(self, key, value):
        if key in self._items:
            self._items.pop(key)
        self._items[key] = value
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def clear(self):
        self._items.clear()


class QtInteractivePlotter(QtWidgets.QMainWindow):
    """Single-window Qt session that owns all detected regions."""

    def __init__(self, all_pt_files, all_data_folder_path, force=False,
                 preview_max_size=1024, parent=None):
        super().__init__(parent)
        self.all_pt_files = list(all_pt_files or [])
        self.all_data_folder_path = all_data_folder_path
        self.force = force
        self.preview_max_size = max(64, int(preview_max_size or 1024))

        self.current_index = -1
        self.current_file_path = None
        self.current_region = None
        self.save_dir = None
        self.pt_file_dir = None
        self.pt_filename = None

        self.optimizable_tensors = None
        self.full_data = None
        self.preview_data = None
        self.preview_stride = 1
        self.preview_probe_dx = None
        self.probe_dx = None
        self.pos_scan_affine = None
        self.saved_params = {}

        self.layer_cache = LRUCache(32)
        self.transform_cache = LRUCache(32)
        self.fft_cache = LRUCache(16)
        self._loading_controls = False

        pg.setConfigOptions(imageAxisOrder='row-major')

        self.setWindowTitle("plot_ptyrad Qt Preview")
        self.setWindowIcon(_create_app_icon())
        self.resize(1500, 950)
        self._build_ui()
        self._load_next_available_region(0)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        central.setObjectName("root")
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        left_pane = QtWidgets.QWidget()
        left_pane.setObjectName("leftPane")
        left_layout = QtWidgets.QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        main_layout.addWidget(left_pane, stretch=1)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setObjectName("viewer")
        self.graphics.setBackground('#ffffff')
        self.plot_item = self.graphics.addPlot(row=0, col=0)
        self.plot_item.getViewBox().setBackgroundColor('#ffffff')
        self.plot_item.setAspectLocked(True)
        self.plot_item.invertY(True)
        self.plot_item.hideAxis('bottom')
        self.plot_item.hideAxis('left')
        self.plot_item.showGrid(x=False, y=False)
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        left_layout.addWidget(self.graphics, stretch=1)

        panel = QtWidgets.QWidget()
        panel.setObjectName("sidePanel")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 18, 18, 18)
        panel_layout.setSpacing(12)

        panel_scroll = QtWidgets.QScrollArea()
        panel_scroll.setObjectName("sidePanelScroll")
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel_scroll.setWidget(panel)
        panel_scroll.setMinimumWidth(450)
        panel_scroll.setMaximumWidth(520)
        panel_scroll.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        main_layout.addWidget(panel_scroll)

        title = QtWidgets.QLabel("PtyRAD Qt Viewer")
        title.setObjectName("panelTitle")
        subtitle = QtWidgets.QLabel("Fast preview controls with full-resolution export")
        subtitle.setObjectName("panelSubtitle")
        panel_layout.addWidget(title)
        panel_layout.addWidget(subtitle)

        self.step_buttons = []
        controls_group = self._make_group_box("View Controls")
        controls_layout = QtWidgets.QGridLayout(controls_group)
        controls_layout.setContentsMargins(10, 14, 10, 10)
        controls_layout.setHorizontalSpacing(14)
        controls_layout.setVerticalSpacing(8)
        left_layout.addWidget(controls_group, stretch=0)

        self.start_spin, self.start_slider, start_row = self._make_int_control("Layer Start", 0, 0, 0)
        self.count_spin, self.count_slider, count_row = self._make_int_control("Layer Count", 1, 1, 1)
        self.rotation_spin, self.rotation_slider, rotation_row = self._make_double_control(
            "Rotation", -180.0, 180.0, 0.0, 0.1, slider_scale=10
        )
        self.crop_x_spin, self.crop_x_slider, crop_x_row = self._make_int_control("Crop X", 0, 0, 0)
        self.crop_y_spin, self.crop_y_slider, crop_y_row = self._make_int_control("Crop Y", 0, 0, 0)
        self.crop_center_x_spin, self.crop_center_x_slider, center_x_row = self._make_int_control("Center X", 0, 0, 0)
        self.crop_center_y_spin, self.crop_center_y_slider, center_y_row = self._make_int_control("Center Y", 0, 0, 0)

        control_rows = (
            start_row,
            count_row,
            rotation_row,
            crop_x_row,
            crop_y_row,
            center_x_row,
            center_y_row,
        )
        for index, row in enumerate(control_rows):
            controls_layout.addWidget(row, index // 3, index % 3)

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(COLORMAP_OPTIONS)
        self.display_combo = QtWidgets.QComboBox()
        self.display_combo.addItems(DISPLAY_MODE_OPTIONS)
        self.gamma_spin, self.gamma_slider, gamma_row = self._make_double_control(
            "FFT Gamma", 0.0, 1.0, 0.0, 0.01, slider_scale=100
        )

        display_group = self._make_group_box("Display")
        display_layout = QtWidgets.QVBoxLayout(display_group)
        display_layout.setSpacing(10)
        display_layout.addWidget(self._make_combo_row("Display Mode", self.display_combo))
        display_layout.addWidget(gamma_row)
        display_layout.addWidget(self._make_combo_row("Colormap", self.cmap_combo))
        panel_layout.addWidget(display_group)

        self.save_image_button = QtWidgets.QPushButton("Save Current Image")
        self.save_video_button = QtWidgets.QPushButton("Save Current Video")
        self.save_mat_button = QtWidgets.QPushButton("Save Current MAT")
        self.save_all_button = QtWidgets.QPushButton("Save All Regions Videos & MAT")
        self.next_button = QtWidgets.QPushButton("Next Region")
        self.end_button = QtWidgets.QPushButton("End")

        actions_group = self._make_group_box("Actions")
        actions_layout = QtWidgets.QGridLayout(actions_group)
        actions_layout.setSpacing(8)
        panel_layout.addWidget(actions_group)

        self.save_image_button.setProperty("primary", True)
        self.next_button.setProperty("accent", True)
        self.end_button.setProperty("danger", True)

        action_buttons = (
            self.save_image_button,
            self.save_video_button,
            self.save_mat_button,
            self.save_all_button,
            self.next_button,
            self.end_button,
        )
        for button in action_buttons:
            button.setMinimumHeight(34)

        actions_layout.addWidget(self.save_image_button, 0, 0, 1, 2)
        actions_layout.addWidget(self.save_video_button, 1, 0)
        actions_layout.addWidget(self.save_mat_button, 1, 1)
        actions_layout.addWidget(self.save_all_button, 2, 0, 1, 2)
        actions_layout.addWidget(self.next_button, 3, 0)
        actions_layout.addWidget(self.end_button, 3, 1)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.param_label = QtWidgets.QLabel("")
        self.param_label.setObjectName("paramSummary")
        self.param_label.setWordWrap(True)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)

        diagnostics_group = self._make_group_box("Diagnostics")
        diagnostics_layout = QtWidgets.QVBoxLayout(diagnostics_group)
        diagnostics_layout.setSpacing(8)
        diagnostics_layout.addWidget(self.progress_bar)
        diagnostics_layout.addWidget(self.param_label)
        diagnostics_layout.addWidget(self.log_box, stretch=1)
        panel_layout.addWidget(diagnostics_group, stretch=1)

        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(75)
        self.update_timer.timeout.connect(self.update_preview)

        controls = (
            self.start_spin,
            self.count_spin,
            self.rotation_spin,
            self.crop_x_spin,
            self.crop_y_spin,
            self.crop_center_x_spin,
            self.crop_center_y_spin,
            self.gamma_spin,
        )
        for control in controls:
            control.valueChanged.connect(self.schedule_update)
        self.cmap_combo.currentTextChanged.connect(self.schedule_update)
        self.display_combo.currentTextChanged.connect(self.schedule_update)

        self.save_image_button.clicked.connect(self.save_current_image)
        self.save_video_button.clicked.connect(self.save_current_video)
        self.save_mat_button.clicked.connect(self.save_current_mat)
        self.save_all_button.clicked.connect(self.save_all_regions_placeholder)
        self.next_button.clicked.connect(self.next_region)
        self.end_button.clicked.connect(self.end_processing)
        self._apply_style()

    def _make_group_box(self, title):
        group_box = QtWidgets.QGroupBox(title)
        group_box.setObjectName("controlGroup")
        return group_box

    def _make_control_row(self, title, value_widget, slider, stepper):
        row = QtWidgets.QWidget()
        row.setObjectName("controlRow")
        row.setMinimumHeight(68)
        row.setMinimumWidth(250)
        row.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        layout = QtWidgets.QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        label = QtWidgets.QLabel(title)
        label.setObjectName("controlLabel")
        label.setMinimumHeight(24)
        header.addWidget(label)
        header.addStretch(1)
        header.addWidget(value_widget)
        header.addWidget(stepper)

        layout.addLayout(header)
        slider.setMinimumHeight(30)
        slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        layout.addWidget(slider)
        return row

    def _make_int_control(self, title, minimum, maximum, value):
        spinbox = self._make_spinbox(minimum, maximum, value)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setTracking(True)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, (maximum - minimum) // 20))
        slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)
        return spinbox, slider, self._make_control_row(title, spinbox, slider, self._make_stepper(spinbox))

    def _make_double_control(self, title, minimum, maximum, value, step, slider_scale):
        spinbox = self._make_double_spinbox(minimum, maximum, value, step)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(round(minimum * slider_scale)), int(round(maximum * slider_scale)))
        slider.setValue(int(round(value * slider_scale)))
        slider.setTracking(True)
        slider.setSingleStep(max(1, int(round(step * slider_scale))))
        slider.setPageStep(max(1, int(round((maximum - minimum) * slider_scale / 20))))
        slider.setFocusPolicy(QtCore.Qt.StrongFocus)

        def slider_to_spinbox(slider_value):
            spinbox.setValue(slider_value / slider_scale)

        def spinbox_to_slider(spinbox_value):
            slider.setValue(int(round(spinbox_value * slider_scale)))

        slider.valueChanged.connect(slider_to_spinbox)
        spinbox.valueChanged.connect(spinbox_to_slider)
        return spinbox, slider, self._make_control_row(title, spinbox, slider, self._make_stepper(spinbox))

    def _make_stepper(self, spinbox):
        stepper = QtWidgets.QWidget()
        stepper.setObjectName("stepper")
        stepper.setFixedSize(32, 32)
        layout = QtWidgets.QVBoxLayout(stepper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        up_button = QtWidgets.QToolButton()
        down_button = QtWidgets.QToolButton()
        up_button.setObjectName("stepUpButton")
        down_button.setObjectName("stepDownButton")
        up_button.setText("▲")
        down_button.setText("▼")
        up_button.setToolTip("Increase value")
        down_button.setToolTip("Decrease value")

        for button in (up_button, down_button):
            button.setFixedSize(32, 15)
            button.setFocusPolicy(QtCore.Qt.NoFocus)
            button.setAutoRepeat(True)
            button.setAutoRepeatDelay(320)
            button.setAutoRepeatInterval(65)
            self.step_buttons.append(button)

        up_button.clicked.connect(spinbox.stepUp)
        down_button.clicked.connect(spinbox.stepDown)
        layout.addWidget(up_button)
        layout.addWidget(down_button)
        return stepper

    def _make_combo_row(self, title, combo):
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(title)
        label.setObjectName("controlLabel")
        layout.addWidget(label)
        layout.addWidget(combo, stretch=1)
        return row

    def _make_spinbox(self, minimum, maximum, value):
        spinbox = QtWidgets.QSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        spinbox.setAlignment(QtCore.Qt.AlignCenter)
        spinbox.setFixedWidth(92)
        spinbox.setFixedHeight(32)
        return spinbox

    def _make_double_spinbox(self, minimum, maximum, value, step):
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(2)
        spinbox.setValue(value)
        spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        spinbox.setAlignment(QtCore.Qt.AlignCenter)
        spinbox.setFixedWidth(92)
        spinbox.setFixedHeight(32)
        return spinbox

    def _set_int_control_range(self, spinbox, slider, minimum, maximum):
        spinbox.setRange(minimum, maximum)
        slider.setRange(minimum, maximum)

    def _set_double_control_range(self, spinbox, slider, minimum, maximum, slider_scale):
        spinbox.setRange(minimum, maximum)
        slider.setRange(int(round(minimum * slider_scale)), int(round(maximum * slider_scale)))

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget#root {
                background: #dfe7eb;
                color: #1f2a33;
                font-family: Arial, "Microsoft YaHei UI", sans-serif;
                font-size: 14px;
            }
            QWidget#viewer {
                border: 1px solid #c9d5dd;
                border-radius: 8px;
                background: #ffffff;
            }
            QWidget#sidePanel {
                background: #f7f9fb;
                border: 1px solid #c9d5dd;
                border-radius: 8px;
            }
            QScrollArea#sidePanelScroll {
                background: transparent;
                border: none;
            }
            QLabel#panelTitle {
                color: #10202b;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#panelSubtitle {
                color: #5f7280;
                font-size: 13px;
                padding-bottom: 4px;
            }
            QGroupBox#controlGroup {
                background: #ffffff;
                border: 1px solid #d6e0e6;
                border-radius: 7px;
                margin-top: 10px;
                padding: 12px 10px 10px 10px;
                font-weight: 700;
                font-size: 14px;
                color: #20313d;
            }
            QGroupBox#controlGroup::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                background: #ffffff;
            }
            QLabel#controlLabel {
                color: #314552;
                font-weight: 600;
                font-size: 14px;
                padding-top: 1px;
                padding-bottom: 3px;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background: #f9fbfc;
                border: 1px solid #cbd8df;
                border-radius: 5px;
                min-height: 30px;
                padding: 2px 6px;
                color: #17242d;
                font-size: 14px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #2f80ed;
                background: #ffffff;
            }
            QWidget#stepper {
                background: transparent;
            }
            QToolButton#stepUpButton, QToolButton#stepDownButton {
                background: #eef5f8;
                border: 1px solid #c4d1d9;
                color: #0f2f45;
                font-size: 14px;
                font-weight: 800;
                padding: 0;
            }
            QToolButton#stepUpButton {
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QToolButton#stepDownButton {
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
            }
            QToolButton#stepUpButton:hover, QToolButton#stepDownButton:hover {
                background: #dfeaf0;
                border-color: #2f80ed;
                color: #1d4ed8;
            }
            QToolButton#stepUpButton:pressed, QToolButton#stepDownButton:pressed {
                background: #cfe0ea;
            }
            QSlider::groove:horizontal {
                height: 7px;
                background: #d8e2e8;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #2f80ed;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
                background: #ffffff;
                border: 2px solid #2f80ed;
            }
            QPushButton {
                background: #e8eef2;
                border: 1px solid #cbd8df;
                border-radius: 6px;
                color: #1f2f3a;
                font-weight: 600;
                font-size: 14px;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background: #dce8ef;
            }
            QPushButton[primary="true"] {
                background: #2563eb;
                border-color: #1d4ed8;
                color: #ffffff;
            }
            QPushButton[primary="true"]:hover {
                background: #1d4ed8;
            }
            QPushButton[accent="true"] {
                background: #f59e0b;
                border-color: #d97706;
                color: #ffffff;
            }
            QPushButton[danger="true"] {
                background: #e11d48;
                border-color: #be123c;
                color: #ffffff;
            }
            QProgressBar {
                border: 1px solid #cbd8df;
                border-radius: 5px;
                height: 14px;
                text-align: center;
                background: #eef4f7;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: #2f80ed;
            }
            QLabel#paramSummary {
                background: #f5f8fa;
                border: 1px solid #d6e0e6;
                border-radius: 6px;
                color: #314552;
                font-size: 13px;
                padding: 8px;
            }
            QPlainTextEdit#logBox {
                background: #111b24;
                border: 1px solid #263847;
                border-radius: 6px;
                color: #d5e3ec;
                font-family: Consolas, "Cascadia Mono", monospace;
                font-size: 13px;
                padding: 6px;
            }
        """)

    def _set_controls_enabled(self, enabled):
        for widget in (
            self.start_spin,
            self.start_slider,
            self.count_spin,
            self.count_slider,
            self.rotation_spin,
            self.rotation_slider,
            self.crop_x_spin,
            self.crop_x_slider,
            self.crop_y_spin,
            self.crop_y_slider,
            self.crop_center_x_spin,
            self.crop_center_x_slider,
            self.crop_center_y_spin,
            self.crop_center_y_slider,
            self.gamma_spin,
            self.gamma_slider,
            self.cmap_combo,
            self.display_combo,
            self.save_image_button,
            self.save_video_button,
            self.save_mat_button,
            self.save_all_button,
            self.next_button,
            *self.step_buttons,
        ):
            widget.setEnabled(enabled)

    def log(self, message):
        self.log_box.appendPlainText(str(message))

    def schedule_update(self, *_):
        if self._loading_controls or self.preview_data is None:
            return
        self.update_timer.start()

    def _load_next_available_region(self, start_index):
        if not self.all_pt_files:
            self.log("No model files were detected.")
            self._set_controls_enabled(False)
            return False

        index = start_index
        while index < len(self.all_pt_files):
            pt_file_path, region_name = self.all_pt_files[index]
            save_dir = os.path.join(self.all_data_folder_path, 'Data_Saved', region_name)
            os.makedirs(save_dir, exist_ok=True)
            if check_if_processed(save_dir, 'general', force=self.force):
                self.log(f"Skipping processed region {region_name}: {save_dir}")
                index += 1
                continue
            return self._load_region(index)

        self.log("No more unprocessed regions. Use --force to reopen existing outputs.")
        self._set_controls_enabled(False)
        return False

    def _load_region(self, index):
        pt_file_path, region_name = self.all_pt_files[index]
        self.current_index = index
        self.current_file_path = pt_file_path
        self.current_region = region_name
        self.save_dir = os.path.join(self.all_data_folder_path, 'Data_Saved', region_name)
        self.pt_file_dir = os.path.dirname(os.path.abspath(pt_file_path))
        self.pt_filename = os.path.basename(pt_file_path)
        os.makedirs(self.save_dir, exist_ok=True)

        self.log("")
        self.log(f"Loading region {region_name}")
        self.log(f"File: {pt_file_path}")
        self.log(f"Save dir: {self.save_dir}")

        model_data = DataProcessor.load_model_file(pt_file_path)
        self.optimizable_tensors = DataProcessor.extract_optimizable_tensors(model_data)
        if 'objp' not in self.optimizable_tensors:
            self.log("No objp tensor found in this file.")
            return False

        tensor_np = self.optimizable_tensors['objp'].detach().cpu().numpy()
        self.full_data = np.transpose(tensor_np, (2, 3, 1, 0)).mean(axis=-1)
        self.probe_dx, self.pos_scan_affine = DataProcessor.load_yml_params(self.pt_file_dir)
        self.preview_data = self._build_preview_data(self.full_data)
        self.preview_probe_dx = self.probe_dx * self.preview_stride if self.probe_dx is not None else None

        self.saved_params = ParameterManager.load_plot_params(self.save_dir)
        initial_rotation = self.saved_params.get('rotation_angle', 0)
        if initial_rotation == 0 and self.pos_scan_affine is not None and len(self.pos_scan_affine) > 2:
            initial_rotation = -self.pos_scan_affine[2]
        self.saved_params['rotation_angle'] = DataProcessor.normalize_angle(initial_rotation)

        self.layer_cache.clear()
        self.transform_cache.clear()
        self.fft_cache.clear()
        self._configure_controls()
        self._set_controls_enabled(True)
        self.log(
            f"Preview: full={self.full_data.shape}, preview={self.preview_data.shape}, "
            f"stride={self.preview_stride}, scale=1/{self.preview_stride}"
        )
        self.update_preview()
        return True

    def _build_preview_data(self, full_data):
        max_edge = max(full_data.shape[0], full_data.shape[1])
        self.preview_stride = max(1, int(math.ceil(max_edge / float(self.preview_max_size))))
        if self.preview_stride == 1:
            return full_data
        return full_data[::self.preview_stride, ::self.preview_stride, :]

    def _configure_controls(self):
        self._loading_controls = True
        try:
            h, w, layers = self.full_data.shape
            self._set_int_control_range(self.start_spin, self.start_slider, 0, max(0, layers - 1))
            self._set_int_control_range(self.count_spin, self.count_slider, 1, max(1, layers))
            self._set_double_control_range(self.rotation_spin, self.rotation_slider, -180.0, 180.0, 10)
            self._set_int_control_range(self.crop_x_spin, self.crop_x_slider, 0, h // 2)
            self._set_int_control_range(self.crop_y_spin, self.crop_y_slider, 0, w // 2)
            self._set_int_control_range(self.crop_center_x_spin, self.crop_center_x_slider, -(h // 2), h // 2)
            self._set_int_control_range(self.crop_center_y_spin, self.crop_center_y_slider, -(w // 2), w // 2)
            self._set_double_control_range(self.gamma_spin, self.gamma_slider, 0.0, 1.0, 100)

            self.start_spin.setValue(self._clamp_int(self.saved_params.get('start_layer', 0), 0, layers - 1))
            self.count_spin.setValue(self._clamp_int(self.saved_params.get('layer_count', 1), 1, layers))
            self.rotation_spin.setValue(float(self.saved_params.get('rotation_angle', 0)))
            self.crop_x_spin.setValue(self._clamp_int(self.saved_params.get('crop_x', 0), 0, h // 2))
            self.crop_y_spin.setValue(self._clamp_int(self.saved_params.get('crop_y', 0), 0, w // 2))
            self.crop_center_x_spin.setValue(
                self._clamp_int(self.saved_params.get('crop_center_x', 0), -(h // 2), h // 2)
            )
            self.crop_center_y_spin.setValue(
                self._clamp_int(self.saved_params.get('crop_center_y', 0), -(w // 2), w // 2)
            )
            self.gamma_spin.setValue(float(self.saved_params.get('fft_gamma', 0.0)))
            self._set_combo_value(self.display_combo, self.saved_params.get('display_mode', 'original'))
            self._set_combo_value(self.cmap_combo, self.saved_params.get('colormap', 'viridis'))
        finally:
            self._loading_controls = False

    def _clamp_int(self, value, minimum, maximum):
        return max(minimum, min(maximum, int(value)))

    def _set_combo_value(self, combo, value):
        index = combo.findText(str(value))
        if index < 0:
            index = 0
        combo.setCurrentIndex(index)

    def get_current_params(self):
        crop_x = int(self.crop_x_spin.value())
        crop_y = int(self.crop_y_spin.value())
        crop_center_x = max(-crop_x, min(crop_x, int(self.crop_center_x_spin.value())))
        crop_center_y = max(-crop_y, min(crop_y, int(self.crop_center_y_spin.value())))
        return {
            'start_layer': int(self.start_spin.value()),
            'layer_count': int(self.count_spin.value()),
            'rotation_angle': float(self.rotation_spin.value()),
            'crop_x': crop_x,
            'crop_y': crop_y,
            'crop_center_x': crop_center_x,
            'crop_center_y': crop_center_y,
            'colormap': self.cmap_combo.currentText(),
            'display_mode': self.display_combo.currentText(),
            'fft_gamma': float(self.gamma_spin.value()),
        }

    def _preview_params(self, params):
        scaled = dict(params)
        stride = self.preview_stride
        for key in ('crop_x', 'crop_y', 'crop_center_x', 'crop_center_y'):
            scaled[key] = int(round(params.get(key, 0) / stride))
        return scaled

    def update_preview(self):
        if self.preview_data is None:
            return

        params = self.get_current_params()
        preview_params = self._preview_params(params)
        layer_data = self._get_layer_sum(self.preview_data, params['start_layer'], params['layer_count'])
        display_data, extent, cbar_label = self._render_preview(layer_data, preview_params)

        lut = self._colormap_lut(params['colormap'])
        levels = self._levels(display_data)
        self.image_item.setLookupTable(lut)
        self.image_item.setImage(
            np.ascontiguousarray(display_data),
            autoLevels=False,
            levels=levels
        )
        self._apply_extent(extent)

        self.plot_item.setTitle(
            f"{self.current_region} | {params['display_mode']} | {cbar_label} | "
            f"{display_data.shape[0]}x{display_data.shape[1]} preview",
            color='#10202b',
            size='14pt'
        )
        self.param_label.setText(self._format_param_summary(params, display_data.shape))

    def _format_param_summary(self, params, preview_shape):
        return (
            f"Region: {self.current_region}\n"
            f"File: {self.pt_filename}\n"
            f"Preview: {preview_shape[0]}x{preview_shape[1]} at 1/{self.preview_stride}\n"
            f"Save: {self.save_dir}\n"
            f"Layer: {params['start_layer']} + {params['layer_count']} | "
            f"Rot: {params['rotation_angle']:.1f} | "
            f"Crop: {params['crop_x']},{params['crop_y']} | "
            f"Center: {params['crop_center_x']},{params['crop_center_y']}\n"
            f"Mode: {params['display_mode']} | Gamma: {params['fft_gamma']:.2f} | "
            f"CMap: {params['colormap']}"
        )

    def _get_layer_sum(self, data, start_layer, layer_count):
        start_layer = int(start_layer)
        layer_count = int(layer_count)
        key = (id(data), start_layer, layer_count)
        cached = self.layer_cache.get(key)
        if cached is not None:
            return cached

        end_layer = min(start_layer + layer_count - 1, data.shape[2] - 1)
        if layer_count == 1:
            result = data[:, :, start_layer]
        else:
            result = data[:, :, start_layer:end_layer + 1].sum(axis=2)
        self.layer_cache.set(key, result)
        return result

    def _render_preview(self, layer_data, params):
        rotation_angle = DataProcessor.normalize_angle(params['rotation_angle'])
        transform_key = (
            id(layer_data),
            round(rotation_angle, 3),
            params['crop_x'],
            params['crop_y'],
            params.get('crop_center_x', 0),
            params.get('crop_center_y', 0),
        )
        transformed = self.transform_cache.get(transform_key)
        if transformed is None:
            transformed = DataProcessor.apply_transformations(
                layer_data,
                rotation_angle,
                params['crop_x'],
                params['crop_y'],
                params.get('crop_center_x', 0),
                params.get('crop_center_y', 0)
            )
            self.transform_cache.set(transform_key, transformed)

        display_mode = params.get('display_mode', 'original')
        gamma = float(params.get('fft_gamma', 0.0))
        if display_mode == 'fft':
            fft_key = (transform_key, round(gamma, 4), self.preview_probe_dx)
            cached = self.fft_cache.get(fft_key)
            if cached is not None:
                return cached
            display_data, extent = DataProcessor.calculate_fft_data_and_extent(
                transformed, self.preview_probe_dx, gamma
            )
            extent = self._scale_pixel_extent_if_needed(extent)
            cbar_label = DataProcessor.get_labels_and_units(
                self.preview_probe_dx, is_fft=True, gamma=gamma
            )
            result = (display_data, extent, cbar_label)
            self.fft_cache.set(fft_key, result)
            return result

        display_data = transformed
        extent, _ = DataProcessor.calculate_real_space_extent(
            transformed.shape, self.preview_probe_dx
        )
        extent = self._scale_pixel_extent_if_needed(extent)
        cbar_label = DataProcessor.get_labels_and_units(
            self.preview_probe_dx, is_fft=False, gamma=gamma
        )
        return display_data, extent, cbar_label

    def _scale_pixel_extent_if_needed(self, extent):
        if self.probe_dx is None and self.preview_stride > 1:
            return [value * self.preview_stride for value in extent]
        return extent

    def _apply_extent(self, extent):
        x0, x1 = float(extent[0]), float(extent[1])
        y0, y1 = float(extent[2]), float(extent[3])
        left, right = min(x0, x1), max(x0, x1)
        bottom, top = min(y0, y1), max(y0, y1)
        rect = QtCore.QRectF(left, bottom, right - left, top - bottom)
        self.image_item.setRect(rect)
        self.plot_item.setXRange(left, right, padding=0.02)
        self.plot_item.setYRange(bottom, top, padding=0.02)

    def _colormap_lut(self, colormap_name):
        cmap = colormaps.get_cmap(colormap_name)
        rgba = cmap(np.linspace(0.0, 1.0, 256))
        return np.asarray(rgba[:, :3] * 255, dtype=np.ubyte)

    def _levels(self, data):
        finite = np.asarray(data[np.isfinite(data)])
        if finite.size == 0:
            return (0.0, 1.0)
        data_min = float(finite.min())
        data_max = float(finite.max())
        if data_min == data_max:
            data_max = data_min + 1.0
        return (data_min, data_max)

    def _current_full_layer_data(self, params):
        return self._get_layer_sum(self.full_data, params['start_layer'], params['layer_count'])

    def save_current_image(self):
        if self.full_data is None:
            return
        try:
            params = self.get_current_params()
            current_data = self._current_full_layer_data(params)
            display_data, extent, cbar_label = DataProcessor.render_view(
                current_data, params, self.probe_dx
            )

            start_layer = params['start_layer']
            layer_count = params['layer_count']
            end_layer = min(start_layer + layer_count - 1, self.full_data.shape[2] - 1)
            mode_suffix = (
                f"_FFT_gamma{params['fft_gamma']:.2f}"
                if params['display_mode'] == 'fft'
                else ""
            )
            filepath = self._save_image_file(
                display_data,
                params,
                start_layer,
                end_layer,
                layer_count,
                extent,
                cbar_label,
                mode_suffix
            )
            self.log(f"Saved current image: {filepath}")
        except Exception as e:
            self.log(f"Save Current Image failed: {e}")

    def _save_image_file(self, display_data, params, start_layer, end_layer,
                         layer_count, extent, cbar_label, mode_suffix=""):
        fov_info = DataProcessor.calculate_field_of_view(display_data.shape, self.probe_dx)
        layer_info = f"Layer_{start_layer}" if layer_count == 1 else f"Layers_{start_layer}-{end_layer}"
        rotation_angle = DataProcessor.normalize_angle(params['rotation_angle'])
        rotation_info = f"_Rotation_{rotation_angle:.1f}deg" if abs(rotation_angle) > 0.1 else ""
        filename = f"objp_{layer_info}{rotation_info}{fov_info}{mode_suffix}.png"
        filepath = os.path.join(self.save_dir, filename)

        fig = Figure(figsize=(10, 8), facecolor='white')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            display_data,
            cmap=params['colormap'],
            interpolation='bilinear',
            extent=extent
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
        im.set_clim(*self._levels(display_data))
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=14, labelpad=12, color='#34495e')
        fig.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
        return filepath

    def save_current_video(self):
        if self.full_data is None:
            return
        try:
            params = self.get_current_params()
            ParameterManager.save_plot_params(self.save_dir, params)
            generator = VideoGenerator(self.save_dir)
            generator.save_video(
                self.full_data,
                params,
                self.pt_file_dir,
                self.optimizable_tensors
            )
            self.log(f"Saved current video for region {self.current_region}")
        except Exception as e:
            self.log(f"Save Current Video failed: {e}")

    def save_current_mat(self):
        if self.optimizable_tensors is None:
            return
        try:
            ParameterManager.auto_save_mat_file(
                self.save_dir,
                self.optimizable_tensors,
                self.pt_filename
            )
            self.log(f"Saved current MAT for region {self.current_region}")
        except Exception as e:
            self.log(f"Save Current MAT failed: {e}")

    def save_all_regions_placeholder(self):
        self.log("Save All Regions Videos & MAT will be moved to a cancellable background worker in Phase 2.")

    def next_region(self):
        if self.save_dir and self.full_data is not None:
            ParameterManager.save_plot_params(self.save_dir, self.get_current_params())
        self._load_next_available_region(self.current_index + 1)

    def end_processing(self):
        if self.save_dir and self.full_data is not None:
            ParameterManager.save_plot_params(self.save_dir, self.get_current_params())
        PROCESSING_STATE['end_processing'] = True
        self.close()


def run_qt_session(all_pt_files, all_data_folder_path, force=False, preview_max_size=1024):
    """Create one QApplication and run a single Qt session for all files."""
    _set_windows_app_id()
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    app.setWindowIcon(_create_app_icon())

    window = QtInteractivePlotter(
        all_pt_files,
        all_data_folder_path,
        force=force,
        preview_max_size=preview_max_size
    )
    window.show()

    if owns_app:
        return app.exec() == 0
    return True
