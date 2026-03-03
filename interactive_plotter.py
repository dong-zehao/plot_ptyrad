"""
交互式绘图模块
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import PROCESSING_STATE, UI_LAYOUT, MATPLOTLIB_CONFIG
from ui_components import UIComponents
from data_processor import DataProcessor, ParameterManager
from video_generator import VideoGenerator
from file_utils import check_if_processed


class InteractivePlotter:
    """交互式绘图器"""
    
    def __init__(self, objp_tensor, pt_file_dir, save_dir, optimizable_tensors=None, all_pt_files=None):
        self.objp_tensor = objp_tensor
        self.pt_file_dir = pt_file_dir
        self.save_dir = save_dir
        self.optimizable_tensors = optimizable_tensors
        self.all_pt_files = all_pt_files
        
        # 重置状态
        global PROCESSING_STATE
        PROCESSING_STATE['next_region'] = False
        PROCESSING_STATE['end_processing'] = False
        
        # 设置matplotlib样式
        plt.rcParams.update(MATPLOTLIB_CONFIG)
        
        # 初始化数据
        self._prepare_data()
        
    def _prepare_data(self):
        """准备数据"""
        # 读取参数
        self.probe_dx, self.pos_scan_affine = DataProcessor.load_yml_params(self.pt_file_dir)
        self.saved_params = ParameterManager.load_plot_params(self.save_dir)
        
        # 设置初始旋转角度
        self.initial_rotation = self.saved_params.get('rotation_angle', 0)
        if self.initial_rotation == 0 and self.pos_scan_affine is not None and len(self.pos_scan_affine) > 2:
            self.initial_rotation = -self.pos_scan_affine[2]
        self.initial_rotation = DataProcessor.normalize_angle(self.initial_rotation)
        
        # 处理数据
        tensor_np = self.objp_tensor.detach().cpu().numpy()
        self.data = np.transpose(tensor_np, (2, 3, 1, 0)).mean(axis=-1)
        
        # 获取slice_thickness
        self.slice_thickness = None
        if self.optimizable_tensors and 'slice_thickness' in self.optimizable_tensors:
            self.slice_thickness = self.optimizable_tensors['slice_thickness'].detach().cpu().numpy().item()
    
    def create_interface(self):
        """创建交互界面"""
        # 创建图形界面
        self.fig = plt.figure(figsize=UI_LAYOUT['figure_size'])
        self.fig.patch.set_facecolor('#fafafa')
        
        # 主图像区域
        self.ax = plt.axes(UI_LAYOUT['main_plot'])
        self.ax.set_facecolor('#ffffff')
        
        # 初始显示
        self._setup_initial_display()
        
        # 创建UI控件
        ui_components = UIComponents(self.fig, self.data.shape, self.saved_params, self.initial_rotation)
        self.sliders, self.gamma_slider, self.radio_cmap, self.radio_display_mode, self.buttons = ui_components.create_all_components()
        
        # 添加说明文字
        self._add_info_text()
        
        # 创建视频生成器
        self.video_generator = VideoGenerator(self.save_dir)
        
        # 连接事件
        self._connect_events()
        
        # 初始显示
        self.update_display()
        
        plt.show()
    
    def _setup_initial_display(self):
        """设置初始显示"""
        initial_data = self.data[:, :, 0:1].sum(axis=2)
        if abs(self.initial_rotation) > 0.1:
            from scipy.ndimage import rotate
            initial_data = rotate(initial_data, self.initial_rotation, reshape=False, mode='constant', cval=0)
        
        self.im = self.ax.imshow(initial_data, cmap='viridis', interpolation='bilinear')
        self.ax.set_title("Object Phase (objp) Visualization", 
                    fontsize=20, fontweight='bold', pad=25, color='#2c3e50')
        
        # 去掉坐标轴标签和刻度
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # 美化坐标轴
        for spine in self.ax.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1)
        
        # 创建colorbar
        self.cb = plt.colorbar(self.im, ax=self.ax, pad=0.015, aspect=25)
        self.cb.ax.tick_params(labelsize=12, colors='#34495e')
        self.cb.set_label('Phase (radians)', fontsize=14, labelpad=12, color='#34495e')
        self.cb.outline.set_color('#bdc3c7')
        self.cb.outline.set_linewidth(1)
    
    def _add_info_text(self):
        """添加说明文字"""
        plt.figtext(0.07, 0.13, 'Controls Guide:', fontsize=13, fontweight='bold', color='#2c3e50')
        plt.figtext(0.07, 0.10, '• Set start layer and count to sum consecutive layers • Rotate and crop for optimal view', fontsize=11, color='#7f8c8d')
        plt.figtext(0.07, 0.07, '• Crop values define pixels removed from each edge (centered cropping)', fontsize=11, color='#7f8c8d')
        plt.figtext(0.07, 0.04, f'• Data shape: {self.data.shape[0]}×{self.data.shape[1]} pixels, {self.data.shape[2]} layers • Parameters auto-saved', fontsize=11, color='#7f8c8d')
    
    def _connect_events(self):
        """连接事件处理器"""
        for slider in self.sliders.values():
            slider.on_changed(self.update_display)
        self.gamma_slider.on_changed(self.update_display)  # 新增
        self.radio_cmap.on_clicked(self.update_display)
        self.radio_display_mode.on_clicked(self.update_display)  # 新增
        
        # 连接按钮事件
        self.buttons['save_image'].on_clicked(self.save_current_image)
        self.buttons['save_all_videos'].on_clicked(self.save_all_videos_batch)
        self.buttons['next_region'].on_clicked(self.next_region_clicked)
        self.buttons['end'].on_clicked(self.end_processing_clicked)

    def update_display(self, val=None):
        """更新显示"""
        # 获取参数
        start_layer = int(self.sliders['start'].val)
        layer_count = int(self.sliders['count'].val)
        rotation_angle = DataProcessor.normalize_angle(self.sliders['rotation'].val)
        crop_x = int(self.sliders['crop_x'].val)
        crop_y = int(self.sliders['crop_y'].val)
        display_mode = self.radio_display_mode.value_selected
        gamma = self.gamma_slider.val
        
        # 更新旋转滑块（如果角度被规范化）
        if abs(rotation_angle - self.sliders['rotation'].val) > 0.1:
            self.sliders['rotation'].set_val(rotation_angle)
            return
        
        # 处理层数据
        current_data = self._process_layer_data(start_layer, layer_count)
        current_data = DataProcessor.apply_transformations(current_data, rotation_angle, crop_x, crop_y)
        
        # 根据显示模式处理数据
        if display_mode == 'fft':
            display_data, extent = DataProcessor.calculate_fft_data_and_extent(current_data, self.probe_dx, gamma)
        else:
            display_data = current_data
            extent, _ = DataProcessor.calculate_real_space_extent(current_data.shape, self.probe_dx)
        
        cbar_label = DataProcessor.get_labels_and_units(
            self.probe_dx, is_fft=(display_mode == 'fft'), gamma=gamma)
        
        # 更新图像显示
        self._update_image_display(display_data, start_layer, layer_count, rotation_angle, 
                                 crop_x, crop_y, display_mode, extent, cbar_label, gamma)

    def _process_layer_data(self, start_layer, layer_count):
        """处理层数据"""
        end_layer = min(start_layer + layer_count - 1, self.data.shape[2] - 1)
        if start_layer + layer_count > self.data.shape[2]:
            max_count = self.data.shape[2] - start_layer
            self.sliders['count'].set_val(max_count)
            layer_count = max_count
            end_layer = start_layer + layer_count - 1
        
        if layer_count == 1:
            return self.data[:, :, start_layer]
        else:
            return self.data[:, :, start_layer:end_layer+1].sum(axis=2)
    
    def _update_image_display(self, display_data, start_layer, layer_count, rotation_angle, 
                            crop_x, crop_y, display_mode, extent, cbar_label, gamma):
        """更新图像显示"""
        # 更新图像
        self.im.set_data(display_data)
        self.im.set_cmap(self.radio_cmap.value_selected)
        self.im.set_extent(extent)
        
        # 设置坐标轴范围和标签
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal', adjustable='box')
        
        # 设置色阶
        data_min, data_max = display_data.min(), display_data.max()
        self.im.set_clim(data_min, data_max)
        self.cb.update_normal(self.im)
        self.cb.set_label(cbar_label, fontsize=14, labelpad=12, color='#34495e')
        
        # 更新标题
        end_layer = min(start_layer + layer_count - 1, self.data.shape[2] - 1)
        layer_info = f" - Layer {start_layer}" if layer_count == 1 else f" - Layers {start_layer}-{end_layer} ({layer_count} layers summed)"
        rotation_info = f", Rotation: {rotation_angle:.1f}°" if abs(rotation_angle) > 0.1 else ""
        crop_info = f", Crop: {crop_x}×{crop_y}" if crop_x > 0 or crop_y > 0 else ""
        
        if display_mode == 'fft':
            mode_info = f" - FFT (γ={gamma:.2f})"
            title_prefix = "Object Phase FFT"
        else:
            mode_info = ""
            title_prefix = "Object Phase"
        
        range_info = f" [Range: {data_min:.3f} to {data_max:.3f}]"
        
        self.ax.set_title(f"{title_prefix}{layer_info}{rotation_info}{crop_info}{mode_info}{range_info}", 
                    fontsize=12, color='#2c3e50')
        
        self.fig.canvas.draw()
    
    def get_current_params(self):
        """获取当前参数"""
        return {
            'start_layer': int(self.sliders['start'].val),
            'layer_count': int(self.sliders['count'].val),
            'rotation_angle': self.sliders['rotation'].val,
            'crop_x': int(self.sliders['crop_x'].val),
            'crop_y': int(self.sliders['crop_y'].val),
            'colormap': self.radio_cmap.value_selected,
            'display_mode': self.radio_display_mode.value_selected,
            'fft_gamma': self.gamma_slider.val  # 新增
        }
    
    def save_current_image(self, event):
        """保存当前图像"""
        params = self.get_current_params()
        
        # 处理数据
        start_layer = params['start_layer']
        layer_count = params['layer_count']
        end_layer = min(start_layer + layer_count - 1, self.data.shape[2] - 1)
        
        current_data = self._process_layer_data(start_layer, layer_count)
        current_data = DataProcessor.apply_transformations(
            current_data, DataProcessor.normalize_angle(params['rotation_angle']),
            params['crop_x'], params['crop_y'])
        
        # 根据显示模式处理数据
        if params['display_mode'] == 'fft':
            display_data, extent = DataProcessor.calculate_fft_data_and_extent(
                current_data, self.probe_dx, params['fft_gamma'])
            mode_suffix = f"_FFT_gamma{params['fft_gamma']:.2f}"
        else:
            display_data = current_data
            extent, _ = DataProcessor.calculate_real_space_extent(current_data.shape, self.probe_dx)
            mode_suffix = ""
        
        cbar_label = DataProcessor.get_labels_and_units(
            self.probe_dx, is_fft=(params['display_mode'] == 'fft'), gamma=params['fft_gamma'])
        
        # 生成文件名并保存
        self._save_image_file(display_data, params, start_layer, end_layer, layer_count, 
                            extent, cbar_label, mode_suffix)
    
    def _save_image_file(self, display_data, params, start_layer, end_layer, layer_count, 
                       extent, cbar_label, mode_suffix=""):
        """保存图像文件"""
        fov_info = DataProcessor.calculate_field_of_view(display_data.shape, self.probe_dx)
        layer_info = f"Layer_{start_layer}" if layer_count == 1 else f"Layers_{start_layer}-{end_layer}"
        rotation_angle = DataProcessor.normalize_angle(params['rotation_angle'])
        rotation_info = f"_Rotation_{rotation_angle:.1f}deg" if abs(rotation_angle) > 0.1 else ""
        
        filename = f"objp_{layer_info}{rotation_info}{fov_info}{mode_suffix}.png"
        filepath = os.path.join(self.save_dir, filename)
        
        # 保存图像
        save_fig, save_ax = plt.subplots(figsize=(10, 8))
        save_fig.patch.set_facecolor('white')
        
        save_im = save_ax.imshow(display_data, cmap=params['colormap'], 
                               interpolation='bilinear', extent=extent)
        save_ax.set_aspect('equal', adjustable='box')
        
        # 设置坐标轴范围和标签
        save_ax.set_xlim(extent[0], extent[1])
        save_ax.set_ylim(extent[2], extent[3])
        save_ax.set_xticks([])
        save_ax.set_yticks([])
        save_ax.set_xlabel('')
        save_ax.set_ylabel('')
        
        data_min, data_max = display_data.min(), display_data.max()
        save_im.set_clim(data_min, data_max)
        
        save_cb = plt.colorbar(save_im, ax=save_ax, pad=0.02)
        save_cb.set_label(cbar_label, fontsize=14, labelpad=12, color='#34495e')
        save_fig.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(save_fig)
        
        print(f"图像已保存到: {filepath}")
    
    def save_all_videos_batch(self, event):
        """批量保存所有视频"""            
        print("开始批量生成视频...")
        current_params = self.get_current_params()
        ParameterManager.save_plot_params(self.save_dir, current_params)
        
        if not self.all_pt_files:
            print("未找到任何模型文件进行批量处理")
            return
        
        print(f"找到 {len(self.all_pt_files)} 个模型文件需要处理")
        
        # 处理每个文件
        for pt_file_path, region_number in self.all_pt_files:
            self._process_video_for_region(pt_file_path, region_number, current_params)
        
        print("\n批量视频生成完成!")
    
    def _process_video_for_region(self, pt_file_path, region_number, current_params):
        """为单个区域处理视频"""
        try:
            print(f"\n处理文件: {os.path.basename(pt_file_path)} (区域: {region_number})")
            
            # 检查是否已经生成过视频
            region_save_dir = os.path.join(os.path.dirname(self.save_dir), region_number)
            if check_if_processed(region_save_dir, 'video'):
                print(f"  区域 {region_number} 已存在视频文件，跳过...")
                return
            
            region_video_generator = VideoGenerator(region_save_dir)
            
            model_data = DataProcessor.load_model_file(pt_file_path)
            file_optimizable_tensors = DataProcessor.extract_optimizable_tensors(model_data)
            
            # 处理数据并生成视频
            file_tensor_np = file_optimizable_tensors['objp'].detach().cpu().numpy()
            file_data = np.transpose(file_tensor_np, (2, 3, 1, 0)).mean(axis=-1)
            
            file_pt_dir = os.path.dirname(os.path.abspath(pt_file_path))
            region_video_generator.save_video(file_data, current_params, file_pt_dir, 
                                     file_optimizable_tensors)
            
        except Exception as e:
            print(f"处理文件 {os.path.basename(pt_file_path)} 时出错: {e}")
    
    def next_region_clicked(self, event):
        """处理下一个区域"""
        global PROCESSING_STATE
        PROCESSING_STATE['next_region'] = True
        ParameterManager.save_plot_params(self.save_dir, self.get_current_params())
        print("正在跳转到下一个区域...")
        plt.close(self.fig)
    
    def end_processing_clicked(self, event):
        """结束处理"""
        global PROCESSING_STATE
        PROCESSING_STATE['end_processing'] = True
        ParameterManager.save_plot_params(self.save_dir, self.get_current_params())
        print("正在结束处理...")
        plt.close(self.fig)
