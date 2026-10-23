"""Responsive Qt/PyQtGraph viewer with latest-request-only background rendering."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from matplotlib import colormaps

from .config import PROCESSING_STATE
from .data_processor import DataProcessor, ParameterManager
from .render_engine import RenderEngine
from .ui_components import UIComponents


_application = None


class InteractivePlotter:
    """Keep the existing CLI and export workflow with native Qt interaction."""

    def __init__(self, objp_tensor, pt_file_dir, save_dir, optimizable_tensors=None, all_pt_files=None):
        self.objp_tensor = objp_tensor
        self.pt_file_dir = pt_file_dir
        self.save_dir = save_dir
        self.optimizable_tensors = optimizable_tensors
        self.all_pt_files = all_pt_files
        PROCESSING_STATE.update(next_region=False, end_processing=False)
        self.probe_dx, self.pos_scan_affine = DataProcessor.load_yml_params(pt_file_dir)
        self.saved_params = ParameterManager.load_plot_params(save_dir)
        self.initial_rotation = self.saved_params.get('rotation_angle', 0)
        if self.initial_rotation == 0 and self.pos_scan_affine is not None and len(self.pos_scan_affine) > 2:
            self.initial_rotation = -self.pos_scan_affine[2]
        self.initial_rotation = DataProcessor.normalize_angle(self.initial_rotation)
        self.data = DataProcessor.prepare_object_volume(objp_tensor)
        if any(size == 0 for size in self.data.shape):
            raise ValueError('objp 的空间维度和层数必须大于零。')
        self.engine = RenderEngine(self.data, self.probe_dx)
        self._render_future = self._export_future = None
        self._pending_params = None
        self._closed = False
        self._close_action = None
        self.last_result = None
        self._color_maps = {}

    def create_interface(self, block=True):
        """Show the viewer; block per region unless embedded or tested with block=False."""
        global _application
        _application = QtWidgets.QApplication.instance()
        if _application is None:
            _application = QtWidgets.QApplication([])
            _application.setQuitOnLastWindowClosed(False)
        self.window = UIComponents(self.data.shape, self.saved_params, self.initial_rotation)
        self.sliders = self.window.controls
        self.textboxes = {name: control.editor for name, control in self.sliders.items()}
        self.gamma_slider = self.window.gamma
        self.radio_cmap = self.window.colormap
        self.radio_display_mode = self.window.display_mode
        self.buttons = self.window.buttons
        self._render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptyrad-preview')
        self._export_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptyrad-export')
        self._poll_timer = QtCore.QTimer(self.window)
        self._poll_timer.setInterval(16)
        self._poll_timer.timeout.connect(self._poll_workers)
        self._request_timer = QtCore.QTimer(self.window)
        self._request_timer.setSingleShot(True)
        self._request_timer.setInterval(16)
        self._request_timer.timeout.connect(self._start_pending_render)
        self._save_timer = QtCore.QTimer(self.window)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(350)
        self._save_timer.timeout.connect(self._save_params)
        for control in self.sliders.values():
            control.changed.connect(self.update_display)
        self.gamma_slider.changed.connect(self.update_display)
        self.radio_display_mode.changed.connect(self.update_display)
        self.radio_cmap.changed.connect(self._change_colormap)
        self.buttons['save_image'].clicked.connect(self._request_image_export)
        self.buttons['save_all_videos'].clicked.connect(self.save_all_videos_batch)
        self.buttons['next_region'].clicked.connect(self.next_region_clicked)
        self.buttons['end'].clicked.connect(self.end_processing_clicked)
        self.window.can_close = self._can_close
        self.window.closing.connect(self._on_close)
        self._change_colormap()
        self.update_display()
        self.window.show()
        if block:
            loop = QtCore.QEventLoop()
            self.window.closing.connect(loop.quit)
            loop.exec()
            self.window.deleteLater()

    def _change_colormap(self):
        name = self.radio_cmap.value
        if name not in self._color_maps:
            positions = np.linspace(0, 1, 256)
            self._color_maps[name] = pg.ColorMap(positions, colormaps[name](positions, bytes=True))
        self.window.colorbar.setColorMap(self._color_maps[name])
        self._save_timer.start()

    def get_current_params(self):
        """Snapshot valid values using the original parameter names and crop axes."""
        start = int(self.sliders['start'].value)
        count = min(int(self.sliders['count'].value), self.data.shape[2] - start)
        crop_x = int(self.sliders['crop_x'].value)
        crop_y = int(self.sliders['crop_y'].value)
        return {
            'start_layer': start,
            'layer_count': count,
            'rotation_angle': DataProcessor.normalize_angle(self.sliders['rotation'].value),
            'crop_x': crop_x, 'crop_y': crop_y,
            'crop_center_x': max(-crop_x, min(crop_x, int(self.sliders['crop_center_x'].value))),
            'crop_center_y': max(-crop_y, min(crop_y, int(self.sliders['crop_center_y'].value))),
            'colormap': self.radio_cmap.value,
            'display_mode': self.radio_display_mode.value,
            'fft_gamma': self.gamma_slider.value,
        }

    @staticmethod
    def _data_key(params):
        # Color changes are LUT-only. Gamma has no effect in original mode.
        return tuple((key, value) for key, value in params.items()
                     if key != 'colormap' and
                     not (key == 'fft_gamma' and params['display_mode'] == 'original'))

    def update_display(self, *_):
        """Coalesce drag events, keeping at most one running and one latest request."""
        if self._closed:
            return
        count = self.sliders['count']
        blocker = QtCore.QSignalBlocker(count)
        count.set_maximum(self.data.shape[2] - int(self.sliders['start'].value))
        del blocker
        params = self.get_current_params()
        self._save_timer.start()
        self._pending_params = params
        if not self._request_timer.isActive():
            self._request_timer.start()

    def _start_pending_render(self):
        if self._closed or self._render_future is not None or self._pending_params is None:
            return
        params, self._pending_params = self._pending_params, None
        if self.last_result is not None and self._data_key(params) == self._data_key(self.last_result.params):
            return
        self._render_future = self._render_executor.submit(self.engine.render, params)
        self._poll_timer.start()

    def _poll_workers(self):
        if self._render_future is not None and self._render_future.done():
            future, self._render_future = self._render_future, None
            try:
                result = future.result()
                # A finished old frame must never replace newer requested settings.
                if self._data_key(result.params) == self._data_key(self.get_current_params()):
                    self._present(result)
            except Exception as exc:
                self.window.status.setText(f'预览失败：{exc}')
            self._start_pending_render()
        if self._export_future is not None and self._export_future.done():
            future, self._export_future = self._export_future, None
            try:
                message = future.result()
                self.window.status.setText(message)
            except Exception as exc:
                self.window.status.setText(f'导出失败：{exc}')
                print(f'导出失败：{exc}')
            for button in self.buttons.values():
                button.setEnabled(True)
        if self._render_future is None and self._export_future is None:
            self._poll_timer.stop()

    def _present(self, result):
        self.last_result = result
        low, high = result.levels
        # A constant image needs a nonzero display range, without changing its data.
        levels = (low, high) if high > low else (low, low + max(abs(low) * 1e-6, 1e-6))
        self.window.image.setImage(result.data, autoLevels=False, levels=levels)
        x0, x1, y1, y0 = result.extent
        rect = QtCore.QRectF(x0, y0, x1 - x0 or 1, y1 - y0 or 1)
        self.window.image.setRect(rect)
        self.window.plot.setRange(rect, padding=0)
        self.window.colorbar.setLevels(levels)
        self.window.colorbar.axis.setLabel(result.label)
        p = result.params
        start, count = p['start_layer'], p['layer_count']
        layers = f'Layer {start}' if count == 1 else f'Layers {start}–{start + count - 1} ({count} layers summed)'
        title = 'Object Phase FFT' if p['display_mode'] == 'fft' else 'Object Phase'
        title += f' — {layers}'
        if abs(p['rotation_angle']) > 0.1:
            title += f", Rotation: {p['rotation_angle']:.1f}°"
        if p['crop_x'] or p['crop_y']:
            title += f", Crop: {p['crop_x']}×{p['crop_y']} Center: {p['crop_center_x']},{p['crop_center_y']}"
        if p['display_mode'] == 'fft':
            title += f" — FFT (γ={p['fft_gamma']:.2f})"
        self.window.title.setText(f'{title} [Range: {low:.3f} to {high:.3f}]')
        if self._export_future is None:
            self.window.status.setText(f'就绪 · {result.data.shape[1]}×{result.data.shape[0]} · 计算 {result.elapsed_ms:.1f} ms')

    def _save_params(self):
        ParameterManager.save_plot_params(self.save_dir, self.get_current_params())

    def _begin_export(self, operation, params, message):
        if self._export_future is not None:
            return
        self._save_params()
        for button in self.buttons.values():
            button.setEnabled(False)
        self.window.status.setText(message)
        self._export_future = self._export_executor.submit(operation, params)
        self._poll_timer.start()

    def _request_image_export(self, *_):
        self._begin_export(self._export_image, self.get_current_params(), '正在保存图像，仍可调整预览…')

    def save_current_image(self, event=None):
        """Synchronous export for scripts; the UI invokes it on its export worker."""
        return self._export_image(self.get_current_params())

    def _export_image(self, params):
        from .image_export import save_image
        # Independent engine: never race the preview cache or depend on its last frame.
        result = RenderEngine(self.data, self.probe_dx).render(params)
        path = save_image(result, self.save_dir, self.probe_dx)
        return f'图像已保存到: {path}'

    def save_all_videos_batch(self, event=None):
        """Export a parameter snapshot in the background while preview stays usable."""
        if not self.all_pt_files:
            self.window.status.setText('未找到任何模型文件进行批量处理。')
            return
        self._begin_export(self._export_batch, self.get_current_params(), '正在导出视频和 MAT，仍可调整预览…')

    def _export_batch(self, params):
        from .video_generator import batch_save_videos_and_mat
        failures = batch_save_videos_and_mat(self.all_pt_files, params, os.path.dirname(self.save_dir))
        if failures:
            raise RuntimeError('; '.join(failures))
        return '视频和 MAT 导出完成。'

    def next_region_clicked(self, event=None):
        """Save settings and return control to the CLI's next region."""
        if self._export_future is None:
            self._close_action = 'next_region'
            self.window.close()

    def end_processing_clicked(self, event=None):
        """Save settings and stop the CLI loop."""
        if self._export_future is None:
            self._close_action = 'end_processing'
            self.window.close()

    def _can_close(self):
        if self._export_future is not None:
            self.window.status.setText('正在导出，请完成后再关闭窗口。预览仍可使用。')
            return False
        return True

    def _on_close(self):
        if self._closed:
            return
        self._closed = True
        self._save_params()
        # Closing the title bar follows the original behavior: advance to the next region.
        PROCESSING_STATE[self._close_action or 'next_region'] = True
        self._request_timer.stop()
        self._poll_timer.stop()
        self._save_timer.stop()
        self._pending_params = None
        self._render_executor.shutdown(wait=False)
        self._export_executor.shutdown(wait=False)
