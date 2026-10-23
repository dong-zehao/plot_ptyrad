"""Full-resolution PNG output on an offscreen Agg canvas, independent of Qt."""

import os

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .config import MATPLOTLIB_CONFIG
from .data_processor import DataProcessor


def save_image(result, save_dir, probe_dx=None):
    """Save the existing 600 dpi PNG format from an exact parameter snapshot."""
    params = result.params
    start, count = params['start_layer'], params['layer_count']
    layer_info = f'Layer_{start}' if count == 1 else f'Layers_{start}-{start + count - 1}'
    angle = params['rotation_angle']
    rotation_info = f'_Rotation_{angle:.1f}deg' if abs(angle) > 0.1 else ''
    fov = DataProcessor.calculate_field_of_view(result.data.shape, probe_dx)
    mode = f"_FFT_gamma{params['fft_gamma']:.2f}" if params['display_mode'] == 'fft' else ''
    path = os.path.join(save_dir, f'objp_{layer_info}{rotation_info}{fov}{mode}.png')
    os.makedirs(save_dir, exist_ok=True)
    fig = Figure(figsize=(10, 8), facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    image = ax.imshow(result.data, cmap=params['colormap'], interpolation='bilinear', extent=result.extent)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(result.extent[:2])
    ax.set_ylim(result.extent[2:])
    ax.set_xticks([])
    ax.set_yticks([])
    image.set_clim(*result.levels)
    cb = fig.colorbar(image, ax=ax, pad=0.02)
    cb.ax.tick_params(labelsize=MATPLOTLIB_CONFIG['ytick.labelsize'])
    cb.set_label(result.label, fontsize=14, labelpad=12, color='#34495e')
    try:
        fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
    finally:
        fig.clear()
    print(f'图像已保存到: {path}')
    return path
