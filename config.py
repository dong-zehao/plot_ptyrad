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
    'colormap': 'hot'
}

# 颜色映射选项
COLORMAP_OPTIONS = ('viridis', 'plasma', 'inferno', 'magma', 'gray', 'hot', 'jet', 'turbo')

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
    'slider_width': 0.13,
    'colormap_panel': [0.78, 0.35, 0.19, 0.4],
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
