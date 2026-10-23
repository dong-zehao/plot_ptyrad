"""Native Qt controls arranged at the original Matplotlib panel positions."""

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .config import UI_LAYOUT, COLORMAP_OPTIONS, DISPLAY_MODE_OPTIONS


class NumericControl(QtWidgets.QWidget):
    """A native slider and editable value with one change notification."""

    changed = QtCore.Signal()

    def __init__(self, title, minimum, maximum, value, decimals=0, color='#3498db', parent=None):
        super().__init__(parent)
        self.scale = 10 ** decimals
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        label = QtWidgets.QLabel(title)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet('font-weight: bold;')
        layout.addWidget(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(round(minimum * self.scale), round(maximum * self.scale))
        self.slider.setStyleSheet(
            'QSlider::groove:horizontal { height: 5px; background: #dce2e7; border-radius: 2px; }'
            f'QSlider::sub-page:horizontal {{ background: {color}; }}'
            f'QSlider::handle:horizontal {{ background: {color}; width: 12px; margin: -4px 0; border-radius: 6px; }}')
        self.editor = QtWidgets.QDoubleSpinBox() if decimals else QtWidgets.QSpinBox()
        if decimals:
            self.editor.setDecimals(decimals)
            self.editor.setSingleStep(1 / self.scale)
        self.editor.setRange(minimum, maximum)
        self.editor.setKeyboardTracking(False)
        self.editor.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.editor.setAccessibleName(title)
        self.slider.setAccessibleName(title)
        layout.addWidget(self.slider)
        layout.addWidget(self.editor)
        self.slider.valueChanged.connect(self._from_slider)
        self.editor.valueChanged.connect(self.set_value)
        self.set_value(value)

    @property
    def value(self):
        return self.slider.value() / self.scale

    def _from_slider(self, value):
        blocker = QtCore.QSignalBlocker(self.editor)
        self.editor.setValue(value / self.scale if self.scale > 1 else value)
        del blocker
        self.changed.emit()

    def set_value(self, value):
        """Clamp and synchronize both widgets, including unchanged slider values."""
        self.slider.setValue(round(value * self.scale))
        blocker = QtCore.QSignalBlocker(self.editor)
        self.editor.setValue(self.value if self.scale > 1 else int(self.value))
        del blocker

    def set_maximum(self, value):
        """Update the valid range when the start layer changes."""
        self.editor.setMaximum(value)
        self.slider.setMaximum(round(value * self.scale))


class RadioSelector(QtWidgets.QGroupBox):
    """Exclusive native radio buttons with the existing option names."""

    changed = QtCore.Signal()

    def __init__(self, title, options, selected, parent=None):
        super().__init__(title, parent)
        self.group = QtWidgets.QButtonGroup(self)
        self.options = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 6)
        layout.setSpacing(1)
        for option in options:
            button = QtWidgets.QRadioButton(option)
            self.group.addButton(button)
            self.options[option] = button
            layout.addWidget(button)
        self.options[selected if selected in options else options[0]].setChecked(True)
        self.group.buttonClicked.connect(lambda _: self.changed.emit())

    @property
    def value(self):
        return self.group.checkedButton().text()


class SmoothImageItem(pg.ImageItem):
    """Retain smooth image scaling while using PyQtGraph's fast raster path."""

    def paint(self, painter, *args):
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        super().paint(painter, *args)


class UIComponents(QtWidgets.QWidget):
    """Keep the original left plot / bottom controls / right panels layout."""

    closing = QtCore.Signal()

    def __init__(self, data_shape, saved_params, initial_rotation):
        super().__init__()
        self.setWindowTitle('PtyRAD — Object Phase Visualization')
        self.setMinimumSize(1100, 720)
        self.resize(1440, 880)
        self.setStyleSheet('''
            QWidget { background: #fafafa; color: #2c3e50; font-size: 12px; }
            QGroupBox { border: 1px solid #bdc3c7; border-radius: 4px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QRadioButton { font-size: 14px; font-weight: normal; }
            QSpinBox, QDoubleSpinBox { background: white; border: 1px solid #bdc3c7; border-radius: 3px; min-height: 22px; }
            QPushButton { color: white; font-weight: bold; border-radius: 4px; }
            QPushButton:disabled { background: #b0bec5; }
        ''')
        self.placements = []
        self.can_close = lambda: True
        self.graphics = pg.GraphicsLayoutWidget(parent=self)
        self.graphics.setBackground('white')
        self.plot = self.graphics.addPlot()
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.setMenuEnabled(True)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.image = SmoothImageItem(axisOrder='row-major')
        self.image.setAutoDownsample(True)
        self.plot.addItem(self.image)
        self.colorbar = pg.ColorBarItem(values=(0, 1), interactive=False,
                                       colorMapMenu=False)
        self.colorbar.setImageItem(self.image, insert_in=self.plot)
        self.colorbar.axis.setPen('#34495e')
        self.colorbar.axis.setTextPen('#34495e')
        self._place(self.graphics, UI_LAYOUT['main_plot'])
        self.title = QtWidgets.QLabel('Object Phase Visualization', self)
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title.setWordWrap(True)
        self._place(self.title, [0.05, 0.96, 0.70, 0.035])

        self.controls = {}
        h, w, depth = data_shape
        configs = [
            ('start', 'Layer Start', 0, depth - 1, saved_params.get('start_layer', 0)),
            ('count', 'Layer Count', 1, depth, saved_params.get('layer_count', 1)),
            ('rotation', 'Rotation', -180, 180, initial_rotation),
            # At least one pixel must survive a symmetric crop.
            ('crop_x', 'Crop X', 0, (h - 1) // 2, saved_params.get('crop_x', 0)),
            ('crop_y', 'Crop Y', 0, (w - 1) // 2, saved_params.get('crop_y', 0)),
            ('crop_center_x', 'Center X', -(h // 2), h // 2, saved_params.get('crop_center_x', 0)),
            ('crop_center_y', 'Center Y', -(w // 2), w // 2, saved_params.get('crop_center_y', 0)),
        ]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
        for index, (name, title, low, high, initial) in enumerate(configs):
            control = NumericControl(title, low, high, initial, decimals=1 if name == 'rotation' else 0,
                                     color=colors[index], parent=self)
            self.controls[name] = control
            self._place(control, [0.05 + index * 0.10, 0.175, 0.08, 0.095])

        self.colormap = RadioSelector('Color Schemes', COLORMAP_OPTIONS,
                                      saved_params.get('colormap', 'viridis'), self)
        self.display_mode = RadioSelector('Display Mode', DISPLAY_MODE_OPTIONS,
                                          saved_params.get('display_mode', 'original'), self)
        self._place(self.colormap, UI_LAYOUT['colormap_panel'])
        self._place(self.display_mode, UI_LAYOUT['display_mode_panel'])
        self.gamma = NumericControl('FFT Contrast (γ)', 0, 1, saved_params.get('fft_gamma', 0),
                                    decimals=2, color='#e74c3c', parent=self)
        self._place(self.gamma, [0.79, 0.29, 0.17, 0.085])
        self.buttons = {}
        for name, title, color, y in [
            ('save_image', 'Save Image', '#4caf50', 0.88),
            ('save_all_videos', 'Save All Videos & MAT', '#2196f3', 0.82),
            ('next_region', 'Next Region', '#ff9800', 0.08),
            ('end', 'End', '#f44336', 0.02),
        ]:
            button = QtWidgets.QPushButton(title.replace('&', '&&'), self)
            button.setStyleSheet(f'QPushButton:enabled {{ background: {color}; }}')
            self.buttons[name] = button
            self._place(button, [0.78, y, 0.19, 0.05])

        guide = QtWidgets.QLabel(
            '<b>Controls Guide:</b><br>'
            '• Set start layer and count to sum consecutive layers • Rotate and crop for optimal view<br>'
            '• Center X/Y shifts the crop window from image center • Press Enter in text boxes to apply value<br>'
            f'• Data shape: {h}×{w} pixels, {depth} layers • Parameters auto-saved', self)
        guide.setWordWrap(True)
        self._place(guide, [0.07, 0.035, 0.67, 0.105])
        self.status = QtWidgets.QLabel('', self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet('color: #596b78; font-size: 11px;')
        self._place(self.status, [0.78, 0.16, 0.19, 0.11])

    def _place(self, widget, rect):
        self.placements.append((widget, rect))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for widget, (x, y, width, height) in self.placements:
            widget.setGeometry(round(x * self.width()), round((1 - y - height) * self.height()),
                               round(width * self.width()), round(height * self.height()))

    def closeEvent(self, event):
        if not self.can_close():
            event.ignore()
            return
        self.closing.emit()
        event.accept()
