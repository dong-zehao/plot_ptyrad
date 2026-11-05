from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from plot_ptyrad.config import DEFAULT_PARAMS
from plot_ptyrad.video_generator import VideoGenerator


@pytest.mark.parametrize("mode", ["original", "fft"])
def test_streamed_frames_match_individual_png_exports(tmp_path, monkeypatch, mode):
    data = np.random.default_rng(42).random((32, 48, 4))
    data *= [1, 0.01, 0, 12]
    params = dict(DEFAULT_PARAMS, display_mode=mode, rotation_angle=17, fft_gamma=1)
    generator = VideoGenerator(str(tmp_path))
    frames = []

    class Writer:
        def append_data(self, frame):
            frames.append(frame.copy())

    @contextmanager
    def get_writer(path, **kwargs):
        Path(path).touch()
        yield Writer()

    monkeypatch.setattr(imageio, "get_writer", get_writer)
    generator.save_video(data, params, str(tmp_path), {"slice_thickness": 2})
    assert len(frames) == data.shape[2]
    for index, frame in enumerate(frames):
        fig = generator.generate_frame(data, index, params, slice_thickness=2)
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight",
                        facecolor="white", pad_inches=0.1)
            buffer.seek(0)
            pixels = imageio.imread(buffer)
        fig.clear()
        if index == 0:
            h, w = pixels.shape[:2]
            height, width = h - h % 2, w - w % 2
        expected = np.full((height, width, 4), 255, dtype=np.uint8)
        cropped = pixels[:height, :width]
        expected[:cropped.shape[0], :cropped.shape[1]] = cropped
        np.testing.assert_array_equal(frame, expected)
