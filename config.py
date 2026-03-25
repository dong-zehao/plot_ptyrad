"""
配置文件 - 包含所有常量和默认参数
"""

# 默认参数
DEFAULT_PARAMS = {
    'start_layer': 0,
    'layer_count': 1,
    'rotation_angle': 0,
    'crop_x': 0,
    'crop_y': 0,
    'crop_center_x': 0,
    'crop_center_y': 0,
    'colormap': 'hot',
    'display_mode': 'original',
    'fft_gamma': 0.0  # 新增：FFT gamma参数，0=log scale, 1=linear scale
}

# 颜色映射选项
COLORMAP_OPTIONS = ('viridis', 'inferno', 'magma', 'gray', 'hot')

# 显示模式选项
DISPLAY_MODE_OPTIONS = ('original', 'fft')  # 新增：显示模式选项

# 全局状态控制
PROCESSING_STATE = {
    'next_region': False,
    'end_processing': False
}

# UI布局配置
UI_LAYOUT = {
    'figure_size': (18, 11),
    'main_plot': [0.07, 0.38, 0.68, 0.58],
    'slider_height': 0.028,
    'slider_width': 0.08,
    'colormap_panel': [0.78, 0.40, 0.19, 0.20],  # 进一步调整为gamma滑块腾出空间
    'display_mode_panel': [0.78, 0.66, 0.19, 0.10],
    'gamma_slider_panel': [0.79, 0.30, 0.17, 0.04],  # 新增：gamma滑块位置
    'button_width': 0.19,
    'button_height': 0.05
}

# matplotlib配置
MATPLOTLIB_CONFIG = {
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
}
