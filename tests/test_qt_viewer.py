import json
import threading

import numpy as np
import pytest
from PySide6 import QtCore, QtTest

from plot_ptyrad.config import PROCESSING_STATE
from plot_ptyrad.interactive_plotter import InteractivePlotter


@pytest.fixture
def viewer(tmp_path, qapp, spin_until):
    destination = tmp_path / 'Data_Saved' / 'region'
    destination.mkdir(parents=True)
    data = np.arange(4 * 12 * 16, dtype=np.float32).reshape(1, 4, 12, 16)
    plotter = InteractivePlotter(data, str(tmp_path), str(destination))
    plotter.create_interface(block=False)
    spin_until(lambda: plotter.last_result is not None)
    yield plotter
    spin_until(lambda: plotter._export_future is None)
    plotter.window.close()
    plotter._render_executor.shutdown(wait=True)


def test_controls_and_orientation(viewer, spin_until):
    editor = viewer.textboxes['rotation']
    editor.setFocus()
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, '27.3')
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    assert viewer.sliders['rotation'].value == 27.3
    viewer.sliders['count'].set_value(4)
    viewer.sliders['start'].set_value(3)
    viewer.sliders['crop_x'].set_value(99)
    viewer.sliders['crop_center_x'].set_value(-99)
    p = viewer.get_current_params()
    assert p['layer_count'] == 1
    assert p['crop_x'] == 5
    assert p['crop_center_x'] == -5
    spin_until(lambda: viewer.last_result.params['start_layer'] == 3)
    assert viewer.last_result.data.shape == (2, 16)
    assert viewer.window.image.axisOrder == 'row-major'
    assert viewer.window.plot.getViewBox().state['yInverted']
    np.testing.assert_array_equal(viewer.window.image.image, viewer.last_result.data)
    viewer.radio_display_mode.options['fft'].click()
    spin_until(lambda: viewer.last_result.params['display_mode'] == 'fft')
    viewer.gamma_slider.set_value(.42)
    spin_until(lambda: viewer.last_result.params['fft_gamma'] == .42)
    assert '0.42' in viewer.window.title.text()


def test_colormap_does_not_render(viewer, monkeypatch):
    def unexpected(*args):
        raise AssertionError('colormap should not resubmit image computations')
    monkeypatch.setattr(viewer.engine, 'render', unexpected)
    old_data = viewer.window.image.image
    for option in viewer.radio_cmap.options.values():
        option.click()
    assert viewer.window.image.image is old_data
    assert viewer._render_future is None


def test_latest_request_wins_and_gui_keeps_ticking(viewer, spin_until, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = viewer.engine.render
    calls = []
    def slow_render(params):
        calls.append(params['rotation_angle'])
        entered.set()
        assert release.wait(10)
        return original(params)
    monkeypatch.setattr(viewer.engine, 'render', slow_render)
    viewer.sliders['rotation'].set_value(1)
    spin_until(entered.is_set)
    ticks = []
    timer = QtCore.QTimer(viewer.window)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(1)
    try:
        for angle in range(2, 80):
            viewer.sliders['rotation'].set_value(angle)
        spin_until(lambda: len(ticks) >= 5)
        assert calls == [1]
        assert viewer.last_result.params['rotation_angle'] == 0
    finally:
        release.set()
        timer.stop()
    spin_until(lambda: viewer.last_result.params['rotation_angle'] == 79)
    assert calls == [1, 79]


def test_export_uses_snapshot_and_prevents_unsafe_close(viewer, spin_until, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    snapshots = []
    def export(params):
        snapshots.append(params)
        entered.set()
        assert release.wait(10)
        return 'saved'
    monkeypatch.setattr(viewer, '_export_image', export)
    viewer.buttons['save_image'].click()
    spin_until(entered.is_set)
    try:
        viewer.sliders['rotation'].set_value(35)
        spin_until(lambda: viewer.last_result.params['rotation_angle'] == 35)
        assert snapshots[0]['rotation_angle'] == 0
        assert not viewer.buttons['next_region'].isEnabled()
        viewer.window.close()
        assert viewer.window.isVisible()
    finally:
        release.set()
    spin_until(lambda: viewer._export_future is None)
    assert viewer.buttons['end'].isEnabled()


def test_export_failure_reenables_controls(viewer, spin_until, monkeypatch):
    def fail(params):
        raise OSError('disk full')
    monkeypatch.setattr(viewer, '_export_image', fail)
    viewer.buttons['save_image'].click()
    spin_until(lambda: viewer._export_future is None)
    assert 'disk full' in viewer.window.status.text()
    assert all(button.isEnabled() for button in viewer.buttons.values())


@pytest.mark.parametrize('action', ['next_region', 'end'])
def test_navigation_and_parameter_persistence(viewer, action):
    viewer.sliders['rotation'].set_value(15)
    viewer.buttons[action].click()
    assert not viewer.window.isVisible()
    key = 'next_region' if action == 'next_region' else 'end_processing'
    assert PROCESSING_STATE[key]
    from pathlib import Path
    saved = json.loads((Path(viewer.save_dir).parent / 'plot_params.json').read_text())
    assert saved['rotation_angle'] == 15


def test_single_pixel_region_clamps_saved_values(tmp_path, qapp, spin_until):
    destination = tmp_path / 'Data_Saved' / 'tiny'
    destination.mkdir(parents=True)
    (destination.parent / 'plot_params.json').write_text(json.dumps({
        'start_layer': 50, 'layer_count': 20, 'crop_x': 100, 'crop_y': 100,
        'colormap': 'unknown', 'display_mode': 'unknown'}))
    plotter = InteractivePlotter(np.zeros((1, 1, 1, 1), dtype=np.float32),
                                 str(tmp_path), str(destination))
    plotter.create_interface(block=False)
    try:
        spin_until(lambda: plotter.last_result is not None)
        assert plotter.last_result.data.shape == (1, 1)
        assert plotter.get_current_params()['start_layer'] == 0
        assert plotter.get_current_params()['crop_x'] == 0
        assert plotter.get_current_params()['layer_count'] == 1
        assert plotter.window.image.levels[1] > plotter.window.image.levels[0]
    finally:
        plotter.window.close()
