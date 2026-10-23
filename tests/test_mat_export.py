import os
import shutil
import subprocess
import sys

import h5py
import matplotlib
import numpy as np
import pytest
from scipy.io import loadmat
from scipy.ndimage import rotate
import torch

matplotlib.use("Agg")

from plot_ptyrad.config import DEFAULT_PARAMS
from plot_ptyrad.data_processor import DataProcessor, ParameterManager
from plot_ptyrad.mat_export import save_mat_file
from plot_ptyrad.video_generator import batch_save_videos_and_mat


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "region" / "nested" / "model.hdf5"
    path.parent.mkdir(parents=True)
    objp = np.arange(2 * 3 * 8 * 10, dtype=np.float32).reshape(2, 3, 8, 10)
    with h5py.File(path, "w") as source:
        source["optimizable_tensors/objp"] = objp
        source["optimizable_tensors/probe"] = np.ones((2, 4), dtype=np.complex64) * (1 + 2j)
        source["slice_thickness"] = 2.5
        source["other/deep/tensor"] = np.arange(12).reshape(3, 4)
        source["other-deep-tensor"] = np.array([99], dtype=np.uint16)
        source["object_3d"] = np.array([77])
        source["_hidden"] = np.array([42])
        source["1invalid"] = np.array([43])
        source["label"] = "metadata"
        source.create_dataset("null", dtype="f")
    tensors = DataProcessor.extract_optimizable_tensors(DataProcessor.load_model_file(path))
    return path, objp, tensors


def test_all_hdf5_tensors_and_transformed_volume(model, tmp_path):
    path, objp, tensors = model
    params = dict(DEFAULT_PARAMS, rotation_angle=27, crop_x=1, crop_y=2,
                  crop_center_x=1, crop_center_y=-1, display_mode="fft",
                  start_layer=2, layer_count=1)
    result = loadmat(save_mat_file(tmp_path, tensors, path.name, params, path))
    expected = np.stack([
        rotate(objp.mean(axis=0)[k], 27, reshape=False, mode="constant", cval=0)[0:6, 3:9]
        for k in range(3)
    ], axis=2)
    np.testing.assert_allclose(result["object_3d"], expected)
    assert result["object_3d"].shape == (6, 6, 3)
    mapping = {str(name.item()): str(source.item()) for name, source in result["tensor_paths"]}
    with h5py.File(path) as source:
        assert len(mapping) == 8
        for name, original in mapping.items():
            np.testing.assert_array_equal(result[name].squeeze(), np.asarray(source[original][()]).squeeze())
    assert mapping["objp"] == "optimizable_tensors/objp"
    assert "label" in str(result["skipped_datasets"])


def test_repeat_export_updates_even_with_existing_mat_and_video(model, tmp_path):
    path, objp, _ = model
    destination = tmp_path / "Data_Saved"
    region = destination / "region"
    region.mkdir(parents=True)
    (region / "existing.mp4").touch()
    for crop in (0, 1):
        params = dict(DEFAULT_PARAMS, crop_x=crop)
        batch_save_videos_and_mat([(str(path), "region")], params, str(destination))
        result = loadmat(region / "model.mat")
        assert result["object_3d"].shape == (8 - 2 * crop, 10, 3)


def test_pt_export(tmp_path):
    path = tmp_path / "model.pt"
    objp = torch.arange(120, dtype=torch.float32).reshape(1, 3, 5, 8)
    torch.save({"optimizable_tensors": {"objp": objp, "extra": torch.tensor([4, 5])}}, path)
    tensors = DataProcessor.extract_optimizable_tensors(DataProcessor.load_model_file(path))
    result = loadmat(ParameterManager.auto_save_mat_file(tmp_path, tensors, path.name))
    np.testing.assert_array_equal(result["object_3d"], objp.numpy()[0].transpose(1, 2, 0))
    np.testing.assert_array_equal(result["extra"], [[4, 5]])


def test_failed_write_preserves_previous_file(model, tmp_path, monkeypatch):
    import plot_ptyrad.mat_export as exporter
    path, _, tensors = model
    target = save_mat_file(tmp_path, tensors, path.name, {}, path)
    original = open(target, "rb").read()
    def fail(*args, **kwargs):
        raise RuntimeError("write failed")
    monkeypatch.setattr(exporter, "savemat", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        save_mat_file(tmp_path, tensors, path.name, {}, path)
    assert open(target, "rb").read() == original
    assert list(tmp_path.glob("*.mat")) == [tmp_path / "model.mat"]


@pytest.mark.parametrize('extension', ['.hdf5', '.pt'])
def test_cli_headless_multiple_regions(model, tmp_path, extension):
    path, _, tensors = model
    if extension == '.pt':
        path = path.with_suffix('.pt')
        torch.save({'optimizable_tensors': tensors}, path)
    second_path = tmp_path / 'region02' / 'nested' / path.name
    second_path.parent.mkdir(parents=True)
    shutil.copyfile(path, second_path)
    # Drive the real CLI and its blocking Qt event loop offscreen.
    driver = '''
import runpy
from PySide6 import QtCore, QtWidgets
app = QtWidgets.QApplication([])
app.setQuitOnLastWindowClosed(False)
timer = QtCore.QTimer()
def advance():
    for window in app.topLevelWidgets():
        if (window.isVisible() and hasattr(window, 'buttons')
                and window.image.image is not None):
            print('CLI_UI_READY')
            window.buttons['next_region'].click()
timer.timeout.connect(advance)
timer.start(100)
runpy.run_module('plot_ptyrad', run_name='__main__')
'''
    result = subprocess.run(
        [sys.executable, "-c", driver, "--folder", str(tmp_path),
         "--file", path.name, "--force"],
        capture_output=True, text=True,
        env=dict(os.environ, MPLBACKEND="Agg", QT_QPA_PLATFORM="offscreen"), timeout=60)
    assert result.returncode == 0, result.stderr
    assert "成功处理了 2/2" in result.stdout
    assert result.stdout.count('CLI_UI_READY') == 2


def test_image_video_and_mat_pipeline(model, tmp_path, qapp, spin_until):
    from plot_ptyrad.interactive_plotter import InteractivePlotter

    path, _, tensors = model
    destination = tmp_path / "Data_Saved"
    region = destination / "region"
    region.mkdir(parents=True)
    params = dict(DEFAULT_PARAMS, rotation_angle=15, crop_x=1, crop_y=1)
    ParameterManager.save_plot_params(str(region), params)
    plotter = InteractivePlotter(tensors["objp"], str(path.parent), str(region),
                                 tensors, [(str(path), "region")])
    plotter.create_interface(block=False)
    try:
        spin_until(lambda: plotter.last_result is not None)
        plotter.buttons['save_image'].click()
        spin_until(lambda: plotter._export_future is None, timeout=60)
        plotter.buttons['save_all_videos'].click()
        spin_until(lambda: plotter._export_future is None, timeout=60)
    finally:
        plotter.window.close()
    assert any(p.stat().st_size > 0 for p in region.glob("*.png"))
    assert any(p.stat().st_size > 0 for p in region.glob("*.mp4"))
    assert loadmat(region / "model.mat")["object_3d"].shape == (6, 8, 3)
