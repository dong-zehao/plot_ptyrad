"""Bounded, single-worker preview cache; scientific transforms match exports."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .data_processor import DataProcessor


@dataclass
class RenderResult:
    """A complete immutable-by-convention preview handed to the GUI thread."""

    data: np.ndarray
    extent: list
    label: str
    params: dict
    levels: tuple
    elapsed_ms: float


class RenderEngine:
    """Cache only the latest 2D intermediates, never a second full 3D volume.

    Use each instance on one worker only. Input volume must remain read-only.
    """

    def __init__(self, data, probe_dx=None):
        self.data = data
        self.probe_dx = probe_dx
        self._layer_key = self._rotation_key = self._crop_key = None
        self._display_key = None
        self._layers = self._rotated = self._cropped = self._power = None
        self._display = None

    def render(self, params):
        """Return exact full-resolution data, reusing unchanged processing stages."""
        began = perf_counter()
        params = dict(params)
        start, count = params['start_layer'], params['layer_count']
        layer_key = (start, count)
        if layer_key != self._layer_key:
            layers = self.data[:, :, start:start + count]
            self._layers = layers[:, :, 0] if count == 1 else layers.sum(axis=2)
            self._layer_key = layer_key
            self._rotation_key = None

        rotation_key = (layer_key, params['rotation_angle'])
        if rotation_key != self._rotation_key:
            self._rotated = DataProcessor.apply_transformations(
                self._layers, params['rotation_angle'], 0, 0)
            self._rotation_key = rotation_key
            self._crop_key = None

        crop_key = (rotation_key, params['crop_x'], params['crop_y'],
                    params['crop_center_x'], params['crop_center_y'])
        if crop_key != self._crop_key:
            self._cropped = DataProcessor.apply_transformations(
                self._rotated, 0, *crop_key[1:])
            if not self._cropped.size:
                raise ValueError('裁剪后图像为空，请减小 Crop X/Y。')
            self._crop_key = crop_key
            self._power = None
            self._display_key = None

        is_fft = params['display_mode'] == 'fft'
        gamma = params['fft_gamma']
        display_key = (crop_key, is_fft, gamma if is_fft else None)
        if display_key != self._display_key:
            if is_fft:
                if self._power is None:
                    self._power = DataProcessor.calculate_fft_power(self._cropped)
                display = DataProcessor.apply_fft_gamma(self._power, gamma)
                extent = DataProcessor.calculate_fft_extent(display.shape, self.probe_dx)
            else:
                display = self._cropped
                extent, _ = DataProcessor.calculate_real_space_extent(display.shape, self.probe_dx)
            display = np.ascontiguousarray(display)
            finite = display[np.isfinite(display)]
            levels = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
            self._display = display, extent, levels
            self._display_key = display_key

        display, extent, levels = self._display
        label = DataProcessor.get_labels_and_units(self.probe_dx, is_fft, gamma)
        return RenderResult(display, extent, label, params, levels,
                            (perf_counter() - began) * 1000)
