"""
视频生成模块
"""

import os
import tempfile
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm

from config import MATPLOTLIB_CONFIG
from data_processor import DataProcessor


class VideoGenerator:
    """视频生成工具类"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        # 设置matplotlib样式
        plt.rcParams.update(MATPLOTLIB_CONFIG)
    
    def generate_frame(self, data, layer_idx, params, slice_thickness=None):
        """生成单帧图像"""
        current_data = data[:, :, layer_idx]
        current_data = DataProcessor.apply_transformations(
            current_data, params['rotation_angle'], params['crop_x'], params['crop_y'])
        
        # 计算深度信息
        depth_info = ""
        if slice_thickness is not None:
            depth_nm = layer_idx * slice_thickness / 10.0
            depth_info = f" (Depth: {depth_nm:.1f} nm)"
        
        # 创建图像
        frame_fig, frame_ax = plt.subplots(figsize=(10, 8))
        frame_fig.patch.set_facecolor('white')
        
        frame_im = frame_ax.imshow(current_data, cmap=params['colormap'], interpolation='bilinear')
        frame_ax.set_title(f"Layer {layer_idx}{depth_info}", 
                         fontsize=16, fontweight='bold', color='#2c3e50')
        
        # 设置样式
        frame_ax.set_xticks([])
        frame_ax.set_yticks([])
        frame_ax.set_aspect('equal', adjustable='box')
        
        # 设置色阶范围
        global_min = data.min()
        global_max = data.max()
        frame_im.set_clim(global_min, global_max)
        
        # 添加colorbar
        frame_cb = plt.colorbar(frame_im, ax=frame_ax, shrink=0.8)
        frame_cb.set_label('Phase (radians)', fontsize=12, color='#2c3e50')
        
        return frame_fig
    
    def save_video(self, data, params, pt_file_dir, optimizable_tensors=None):
        """保存视频文件"""
        # 获取参数
        rotation_angle = DataProcessor.normalize_angle(params['rotation_angle'])
        crop_datashape = (data.shape[0] - params['crop_x']*2, data.shape[1] - params['crop_y']*2)
        fov_info = DataProcessor.calculate_field_of_view(crop_datashape, DataProcessor.load_yml_params(pt_file_dir)[0])
        rotation_info = f"_Rotation_{rotation_angle:.1f}deg" if abs(rotation_angle) > 0.1 else ""     
        
        # 获取slice_thickness
        slice_thickness = None
        if optimizable_tensors and 'slice_thickness' in optimizable_tensors:
            slice_thickness = optimizable_tensors['slice_thickness'].detach().cpu().numpy().item()
           
        video_filename = f"objp_layers_video{rotation_info}{fov_info}.mp4"
        video_filepath = os.path.join(self.save_dir, video_filename)
        
        # 生成视频帧
        frames = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            print(f"  生成视频帧...")
            for layer_idx in tqdm(range(data.shape[2]), desc=f"  Video"):
                frame_fig = self.generate_frame(data, layer_idx, params, slice_thickness)
                
                temp_frame_path = os.path.join(temp_dir, f"frame_{layer_idx:04d}.png")
                frame_fig.savefig(temp_frame_path, dpi=150, bbox_inches='tight', 
                                facecolor='white', pad_inches=0.1)
                frames.append(temp_frame_path)
                plt.close(frame_fig)
            
            # 处理视频编码
            self._encode_video(frames, video_filepath)
            print(f"  视频已保存: {video_filepath}")
            
        except Exception as e:
            print(f"  生成视频失败: {e}")
        
        finally:
            # 清理临时文件
            for frame_path in frames:
                try:
                    os.remove(frame_path)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
    
    def _encode_video(self, frames, video_filepath):
        """编码视频文件"""
        if not frames:
            return
            
        # 验证和调整帧尺寸
        first_frame = imageio.imread(frames[0])
        h, w = first_frame.shape[:2]
        
        # 确保尺寸为偶数
        h_crop = h - (h % 2)
        w_crop = w - (w % 2)
        
        try:
            # 主要方法：使用 MP4 格式
            with imageio.get_writer(video_filepath, fps=5, codec='libx264', 
                                  quality=8, macro_block_size=1) as writer:
                for frame_path in frames:
                    image = imageio.imread(frame_path)
                    # 确保图像尺寸为偶数
                    if image.shape[0] != h_crop or image.shape[1] != w_crop:
                        image = image[:h_crop, :w_crop]
                    writer.append_data(image)
                    
        except Exception as e:
            print(f"    MP4编码失败: {e}")
