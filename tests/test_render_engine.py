import numpy as np
import pytest
from scipy.ndimage import rotate

from plot_ptyrad.config import DEFAULT_PARAMS
from plot_ptyrad.data_processor import DataProcessor
from plot_ptyrad.render_engine import RenderEngine


@pytest.mark.parametrize('gamma', [0, .01, .02, .3, .99, 1])
@pytest.mark.parametrize('mode', ['original', 'fft'])
@pytest.mark.parametrize('probe_dx', [None, .4])
def test_preview_matches_original_math(mode, gamma, probe_dx):
    volume = np.random.default_rng(24).normal(size=(17, 21, 5)).astype(np.float32)
    params = dict(DEFAULT_PARAMS, start_layer=1, layer_count=3,
                  rotation_angle=27.3, crop_x=2, crop_y=3, crop_center_x=1,
                  crop_center_y=-2, display_mode=mode, fft_gamma=gamma)
    expected = rotate(volume[:, :, 1:4].sum(axis=2), 27.3,
                      reshape=False, mode='constant', cval=0)[1:14, 5:20]
    if mode == 'fft':
        ny, nx = expected.shape
        y, x = np.ogrid[:ny, :nx]
        window = np.exp(-((x - nx // 2)**2 / (2 * (nx / 6)**2)
                          + (y - ny // 2)**2 / (2 * (ny / 6)**2)))
        power = np.abs(np.fft.fftshift(np.fft.fft2(expected * window)))**2
        if gamma <= .01:
            expected = np.log10(power + 1e-10)
        elif abs(gamma - 1) < .01:
            expected = power
        else:
            expected = ((power - power.min()) / (power.max() - power.min() + 1e-10))**gamma
    result = RenderEngine(volume, probe_dx).render(params)
    np.testing.assert_array_equal(result.data, expected)
    reference, extent, label = DataProcessor.render_view(volume[:, :, 1:4].sum(axis=2), params, probe_dx)
    np.testing.assert_array_equal(result.data, reference)
    assert result.extent == extent
    assert result.label == label
    assert result.data.flags.c_contiguous


def test_cache_invalidation_and_gamma_reuses_fft(monkeypatch):
    engine = RenderEngine(np.random.default_rng(1).random((32, 40, 5)))
    params = dict(DEFAULT_PARAMS, display_mode='fft', rotation_angle=20)
    first = engine.render(params)
    power, rotated = engine._power, engine._rotated
    def unexpected(*args, **kwargs):
        raise AssertionError('expensive computation was repeated')
    with monkeypatch.context() as patch:
        patch.setattr(DataProcessor, 'calculate_fft_power', unexpected)
        patch.setattr(DataProcessor, 'apply_transformations', unexpected)
        recolored = engine.render(dict(params, colormap='gray'))
        assert recolored.data is first.data
        gamma_result = engine.render(dict(params, fft_gamma=.4))
        assert engine._power is power
        assert not np.array_equal(gamma_result.data, first.data)
    cropped = engine.render(dict(params, crop_x=1))
    assert engine._rotated is rotated
    assert engine._power is not power
    assert cropped.data.shape == (30, 40)
    changed = engine.render(dict(params, start_layer=2, layer_count=2))
    expected, _, _ = DataProcessor.render_view(engine.data[:, :, 2:4].sum(axis=2), params)
    np.testing.assert_array_equal(changed.data, expected)


def test_constant_and_single_pixel_data():
    for shape in [(1, 1, 1), (8, 12, 1)]:
        for mode in ('original', 'fft'):
            result = RenderEngine(np.zeros(shape)).render(dict(DEFAULT_PARAMS, display_mode=mode))
            assert result.data.shape == shape[:2]
            assert np.isfinite(result.data).all()
