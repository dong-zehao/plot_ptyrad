"""
数据处理模块 - 处理张量数据和文件操作
"""

import os
import json
import torch
import numpy as np
import yaml
import scipy.io as sio
from scipy.ndimage import rotate
from config import DEFAULT_PARAMS

class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def get_device():
        """获取可用的计算设备"""
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    @staticmethod
    def normalize_angle(angle):
        """将角度规范化到±180°范围内"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    @staticmethod
    def load_pt_file(filepath):
        """加载.pt文件并返回数据"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        device = DataProcessor.get_device()
        data = torch.load(filepath, weights_only=False, map_location=device)
        return data
    
    @staticmethod
    def extract_optimizable_tensors(data):
        """从加载的数据中提取optimizable_tensors"""
        if 'optimizable_tensors' not in data:
            raise KeyError("在 .pt 文件中未找到 'optimizable_tensors' 键")
        
        optimizable_tensors = data['optimizable_tensors']
        return optimizable_tensors
    
    @staticmethod
    def load_yml_params(pt_file_dir):
        """从yml文件加载参数"""
        yml_files = [f for f in os.listdir(pt_file_dir) if f.endswith('.yml')]
        if not yml_files:
            print(f"警告: 在目录 {pt_file_dir} 中未找到.yml文件")
            return None, None
        
        yml_path = os.path.join(pt_file_dir, yml_files[0])
        try:
            with open(yml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            init_params = data.get('init_params', {})
            probe_dx = init_params.get('probe_dx', None)
            pos_scan_affine = init_params.get('pos_scan_affine', None)
            
            print(f"从文件 {yml_files[0]} 读取参数")
            return probe_dx, pos_scan_affine
        except Exception as e:
            print(f"读取yml文件失败: {e}")
            return None, None
    
    @staticmethod
    def calculate_real_space_extent(data_shape, probe_dx):
        """计算实空间的绝对坐标范围"""
        if probe_dx is not None:
            # 将probe_dx从埃转换为纳米
            pixel_size_nm = probe_dx / 10.0
            
            ny, nx = data_shape
            x_max = (nx - 1) * pixel_size_nm / 2.0
            y_max = (ny - 1) * pixel_size_nm / 2.0
            
            # extent格式: [x_min, x_max, y_max, y_min] (注意y轴翻转)
            extent = [-x_max, x_max, y_max, -y_max]
            return extent, 'nm'
        else:
            # 如果没有probe_dx信息，使用像素单位
            ny, nx = data_shape
            extent = [0, nx, ny, 0]
            return extent, 'pixels'
    
    @staticmethod
    def apply_gaussian_window(data):
        """为FFT应用高斯窗口以减少周期性边界伪影"""
        ny, nx = data.shape
        
        # 创建2D高斯窗口
        y_center, x_center = ny // 2, nx // 2
        y, x = np.ogrid[:ny, :nx]
        
        # 设置高斯窗口的标准差为图像尺寸的1/6，这样在边缘处值接近0
        sigma_y = ny / 6.0
        sigma_x = nx / 6.0
        
        # 计算高斯窗口
        gaussian_window = np.exp(-((x - x_center)**2 / (2 * sigma_x**2) + 
                                 (y - y_center)**2 / (2 * sigma_y**2)))
        
        # 应用窗口
        windowed_data = data * gaussian_window
        
        return windowed_data
    
    @staticmethod
    def calculate_fft_data_and_extent(data, probe_dx, gamma=0.0):
        """计算FFT数据和对应的k空间范围，支持gamma调整"""
        # 应用高斯窗口以减少周期性边界伪影
        windowed_data = DataProcessor.apply_gaussian_window(data)
        
        # 计算2D FFT并移到中心
        fft_data = np.fft.fftshift(np.fft.fft2(windowed_data))
        fft_magnitude = np.abs(fft_data)**2
        
        # 应用gamma调整：gamma校正是将图像强度进行幂次变换
        if gamma <= 0.01:
            # 近似log scale
            fft_display = np.log10(fft_magnitude + 1e-10)
        elif abs(gamma - 1.0) < 0.01:
            # gamma=1为线性显示
            fft_display = fft_magnitude
        else:
            # gamma校正：先归一化到[0,1]，然后应用幂次变换
            normalized_data = (fft_magnitude - fft_magnitude.min()) / (fft_magnitude.max() - fft_magnitude.min() + 1e-10)
            fft_display = normalized_data ** gamma
        
        # 计算k空间范围
        if probe_dx is not None:
            # 将probe_dx从埃转换为纳米
            pixel_size_nm = probe_dx / 10.0
            
            # 计算k空间范围 (单位: 1/nm)
            ny, nx = data.shape
            kx_max = 1.0 / (2.0 * pixel_size_nm)  # Nyquist频率
            ky_max = 1.0 / (2.0 * pixel_size_nm)
            
            # 创建k空间坐标轴 (以1/nm为单位)
            kx = np.linspace(-kx_max, kx_max, nx)
            ky = np.linspace(-ky_max, ky_max, ny)
            
            # extent格式: [x_min, x_max, y_max, y_min] (注意y轴翻转)
            extent = [kx[0], kx[-1], ky[-1], ky[0]]
            
            return fft_display, extent
        else:
            # 如果没有probe_dx信息，使用像素单位
            ny, nx = data.shape
            extent = [-nx//2, nx//2, ny//2, -ny//2]
            return fft_display, extent
    
    @staticmethod
    def get_labels_and_units(probe_dx, is_fft=False, gamma=0.0):
        """获取图像的colorbar标签"""
        if is_fft:
            # 根据gamma值调整colorbar标签
            if gamma <= 0.01:
                cbar_label = 'log10|FFT| (a.u.)'
            elif gamma >= 0.99:
                cbar_label = '|FFT| (a.u.)'
            else:
                cbar_label = f'Mixed FFT (γ={gamma:.2f})'
        else:
            cbar_label = 'Phase (radians)'
        
        return cbar_label
    
    @staticmethod
    def apply_transformations(data, rotation_angle, crop_x, crop_y):
        """应用旋转和裁剪变换"""
        # 应用旋转
        if abs(rotation_angle) > 0.1:
            data = rotate(data, rotation_angle, reshape=False, mode='constant', cval=0)
        
        # 应用裁剪
        if crop_x > 0 or crop_y > 0:
            h, w = data.shape
            x_start = max(0, crop_x)
            x_end = min(h, h - crop_x)
            y_start = max(0, crop_y)
            y_end = min(w, w - crop_y)
            data = data[x_start:x_end, y_start:y_end]
        
        return data
    
    @staticmethod
    def calculate_field_of_view(data_shape, probe_dx):
        """计算视场大小"""
        if probe_dx is None:
            return ""
        
        pixels_x, pixels_y = data_shape[0], data_shape[1]
        fov_x_nm = pixels_x * probe_dx / 10.0
        fov_y_nm = pixels_y * probe_dx / 10.0
        return f"_FOV_{fov_x_nm:.1f}x{fov_y_nm:.1f}nm"

class ParameterManager:
    """参数管理器"""
    
    @staticmethod
    def save_plot_params(save_dir, params):
        """保存绘图参数到JSON文件"""
        parent_dir = os.path.dirname(save_dir)
        params_file = os.path.join(parent_dir, 'plot_params.json')
        
        try:
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            print(f"绘图参数已保存到: {params_file}")
        except Exception as e:
            print(f"保存参数失败: {e}")
    
    @staticmethod
    def load_plot_params(save_dir):
        """从JSON文件加载绘图参数"""
        parent_dir = os.path.dirname(save_dir)
        params_file = os.path.join(parent_dir, 'plot_params.json')
        
        if not os.path.exists(params_file):
            return DEFAULT_PARAMS.copy()
        
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            print(f"已加载绘图参数: {params_file}")
            return params
        except Exception as e:
            print(f"加载参数失败: {e}")
            return DEFAULT_PARAMS.copy()
    
    @staticmethod
    def auto_save_mat_file(save_dir, optimizable_tensors, pt_filename=None):
        """自动保存MAT文件（如果不存在）"""
        import glob
        mat_files = glob.glob(os.path.join(save_dir, '*.mat'))
        if mat_files:
            print(f"MAT文件已存在于: {save_dir}")
            return
        
        if optimizable_tensors is None:
            print("警告: 没有可用的optimizable_tensors数据")
            return
        
        try:
            mat_data = {}
            for name, tensor in optimizable_tensors.items():
                if isinstance(tensor, torch.Tensor):
                    numpy_array = tensor.detach().cpu().numpy()
                    mat_data[name] = numpy_array
                else:
                    mat_data[name] = tensor
            
            mat_filename = pt_filename.replace('.pt', '.mat')
            mat_filepath = os.path.join(save_dir, mat_filename)
            sio.savemat(mat_filepath, mat_data, format='5', do_compression=True)
            print(f"MAT文件已保存到: {mat_filepath}")
        except Exception as e:
            print(f"保存MAT文件失败: {e}")
