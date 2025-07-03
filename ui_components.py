"""
UI组件类 - 管理界面元素的创建和交互
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from config import UI_LAYOUT, COLORMAP_OPTIONS, DISPLAY_MODE_OPTIONS

class UIComponents:
    """UI组件管理器"""
    
    def __init__(self, fig, data_shape, saved_params, initial_rotation):
        self.fig = fig
        self.data_shape = data_shape
        self.saved_params = saved_params
        self.initial_rotation = initial_rotation
        
        self.sliders = {}
        self.gamma_slider = None  # 新增：gamma滑块
        self.radio_cmap = None
        self.radio_display_mode = None  # 新增：显示模式选择器
        self.buttons = {}
        
    def create_all_components(self):
        """创建所有UI组件"""
        self.sliders = self._create_sliders()
        self.gamma_slider = self._create_gamma_slider()  # 新增
        self.radio_cmap = self._create_colormap_selector()
        self.radio_display_mode = self._create_display_mode_selector()  # 新增
        self.buttons = self._create_control_buttons()
        return self.sliders, self.gamma_slider, self.radio_cmap, self.radio_display_mode, self.buttons  # 修改返回值
    
    def _create_sliders(self):
        """创建控制滑块"""
        layout = UI_LAYOUT
        col_positions = [0.05, 0.23, 0.41, 0.59, 0.77]
        base_y = 0.22
        title_y = base_y + 0.035
        
        sliders = {}
        slider_configs = [
            ('start', 0, self.data_shape[2]-1, self.saved_params.get('start_layer', 0), 'Layer Start'),
            ('count', 1, self.data_shape[2], self.saved_params.get('layer_count', 1), 'Layer Count'),
            ('rotation', -180, 180, self.initial_rotation, 'Rotation'),
            ('crop_x', 0, self.data_shape[0]//2, self.saved_params.get('crop_x', 0), 'Crop X'),
            ('crop_y', 0, self.data_shape[1]//2, self.saved_params.get('crop_y', 0), 'Crop Y')
        ]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        edge_colors = ['#2980b9', '#27ae60', '#c0392b', '#e67e22', '#8e44ad']
        
        for i, (name, min_val, max_val, init_val, title) in enumerate(slider_configs):
            ax_slider = plt.axes([col_positions[i], base_y, layout['slider_width'], layout['slider_height']])
            ax_slider.set_facecolor('#ecf0f1')
            
            slider = Slider(ax_slider, '', min_val, max_val, 
                           valinit=init_val, valfmt='%d' if name != 'rotation' else '%.1f',
                           facecolor=colors[i], edgecolor=edge_colors[i], alpha=0.8)
            
            self._style_slider(slider)
            sliders[name] = slider
            
            # 添加标题
            plt.figtext(col_positions[i] + 0.065, title_y, title, 
                       fontsize=10, fontweight='bold', ha='center', color='#34495e')
        
        return sliders
    
    def _style_slider(self, slider):
        """设置滑块样式"""
        slider.label.set_fontsize(12)
        slider.label.set_color('#2c3e50')
        slider.valtext.set_fontsize(11)
        slider.valtext.set_color('#2c3e50')
    
    def _create_colormap_selector(self):
        """创建颜色映射选择器"""
        ax_cmap = plt.axes(UI_LAYOUT['colormap_panel'])
        ax_cmap.set_facecolor('#f8f9fa')
        ax_cmap.set_title('Color Schemes', fontsize=15, fontweight='bold', 
                         pad=5, color='#2c3e50')
        
        radio_cmap = RadioButtons(ax_cmap, COLORMAP_OPTIONS,
                                 activecolor='#3498db', 
                                 radio_props={'s': 40, 'edgecolor': "#99B2C1", 'linewidth': 1.5})
        
        # 美化选择器
        for label in radio_cmap.labels:
            label.set_fontsize(14)
            label.set_color('#2c3e50')
            label.set_fontweight('normal')
        
        for spine in ax_cmap.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1.5)
        
        # 设置初始值
        initial_colormap = self.saved_params.get('colormap', 'viridis')
        radio_cmap.set_active(list(COLORMAP_OPTIONS).index(initial_colormap))
        
        return radio_cmap
    
    def _create_display_mode_selector(self):
        """创建显示模式选择器"""
        ax_display = plt.axes(UI_LAYOUT['display_mode_panel'])
        ax_display.set_facecolor('#f8f9fa')
        ax_display.set_title('Display Mode', fontsize=15, fontweight='bold', 
                            pad=5, color='#2c3e50')
        
        radio_display = RadioButtons(ax_display, DISPLAY_MODE_OPTIONS,
                                   activecolor='#e74c3c', 
                                   radio_props={'s': 40, 'edgecolor': "#E74C3C", 'linewidth': 1.5})
        
        # 美化选择器
        for label in radio_display.labels:
            label.set_fontsize(14)
            label.set_color('#2c3e50')
            label.set_fontweight('normal')
        
        for spine in ax_display.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1.5)
        
        # 设置初始值
        initial_display_mode = self.saved_params.get('display_mode', 'original')
        radio_display.set_active(list(DISPLAY_MODE_OPTIONS).index(initial_display_mode))
        
        return radio_display
    
    def _create_gamma_slider(self):
        """创建gamma调整滑块"""
        ax_gamma = plt.axes(UI_LAYOUT['gamma_slider_panel'])
        ax_gamma.set_facecolor('#ecf0f1')
        
        initial_gamma = self.saved_params.get('fft_gamma', 0.0)
        gamma_slider = Slider(ax_gamma, 'FFT γ', 0.0, 1.0, 
                            valinit=initial_gamma, valfmt='%.2f',
                            facecolor='#e74c3c', edgecolor='#c0392b', alpha=0.8)
        
        self._style_slider(gamma_slider)
        
        # 添加标题
        plt.figtext(0.87, 0.35, 'FFT Contrast', 
                   fontsize=10, fontweight='bold', ha='center', color='#34495e')
        
        return gamma_slider
    
    def _create_control_buttons(self):
        """创建控制按钮"""
        layout = UI_LAYOUT
        buttons = {}
        
        button_configs = [
            ('save_image', 'Save Image', '#4caf50', '#45a049', '#e8f5e8', 0.88),
            ('save_all_videos', 'Save All Videos', '#2196f3', '#1976d2', '#e3f2fd', 0.82),
            ('next_region', 'Next Region', '#ff9800', '#f57c00', '#fff3e0', 0.08),
            ('end', 'End', '#f44336', '#d32f2f', '#ffebee', 0.02)
        ]
        
        for name, text, color, hover_color, bg_color, y_pos in button_configs:
            ax_btn = plt.axes([0.78, y_pos, layout['button_width'], layout['button_height']])
            ax_btn.set_facecolor(bg_color)
            
            button = Button(ax_btn, text, color=color, hovercolor=hover_color)
            button.label.set_fontsize(12)
            button.label.set_fontweight('bold')
            button.label.set_color('#ffffff')
            
            buttons[name] = button
        
        return buttons
