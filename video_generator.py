"""
视频生成模块
"""

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm

from config import MATPLOTLIB_CONFIG
from data_processor import DataProcessor, ParameterManager
from file_utils import check_if_processed


class VideoGenerator:
    """视频生成工具类"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        plt.rcParams.update(MATPLOTLIB_CONFIG)

    def generate_frame(self, data, layer_idx, params, slice_thickness=None, probe_dx=None):
        """生成单帧图像"""
        current_data = data[:, :, layer_idx]

        # 使用render_view统一处理变换和显示
        display_data, extent, cbar_label = DataProcessor.render_view(
            current_data, params, probe_dx)

        title_prefix = ""
        if params.get('display_mode') == 'fft':
            gamma = params.get('fft_gamma', 0.0)
            title_prefix = f"FFT (γ={gamma:.2f}) - "

        # 计算深度信息
        depth_info = ""
        if slice_thickness is not None:
            depth_nm = layer_idx * slice_thickness / 10.0
            depth_info = f" (Depth: {depth_nm:.1f} nm)"

        # 创建图像
        frame_fig, frame_ax = plt.subplots(figsize=(10, 8))
        frame_fig.patch.set_facecolor('white')

        frame_im = frame_ax.imshow(display_data, cmap=params['colormap'],
                                 interpolation='bilinear', extent=extent)
        frame_ax.set_title(f"{title_prefix}Layer {layer_idx}{depth_info}",
                         fontsize=16, fontweight='bold', color='#2c3e50')

        frame_ax.set_xticks([])
        frame_ax.set_yticks([])
        frame_ax.set_aspect('equal', adjustable='box')

        # 设置色阶范围
        if params.get('display_mode') == 'fft':
            global_min = display_data.min()
            global_max = display_data.max()
        else:
            global_min = data.min()
            global_max = data.max()

        frame_im.set_clim(global_min, global_max)

        frame_cb = plt.colorbar(frame_im, ax=frame_ax, shrink=0.8)
        frame_cb.set_label(cbar_label, fontsize=12, color='#2c3e50')

        return frame_fig

    def save_video(self, data, params, pt_file_dir, optimizable_tensors=None):
        """保存视频文件"""
        rotation_angle = DataProcessor.normalize_angle(params['rotation_angle'])
        probe_dx, _ = DataProcessor.load_yml_params(pt_file_dir)
        crop_datashape = (data.shape[0] - params['crop_x']*2, data.shape[1] - params['crop_y']*2)
        fov_info = DataProcessor.calculate_field_of_view(crop_datashape, probe_dx)
        rotation_info = f"_Rotation_{rotation_angle:.1f}deg" if abs(rotation_angle) > 0.1 else ""

        if params.get('display_mode') == 'fft':
            gamma = params.get('fft_gamma', 0.0)
            mode_suffix = f"_FFT_gamma{gamma:.2f}"
        else:
            mode_suffix = ""

        slice_thickness = None
        if optimizable_tensors and 'slice_thickness' in optimizable_tensors:
            slice_thickness = optimizable_tensors['slice_thickness'].detach().cpu().numpy().item()

        video_filename = f"objp_layers_video{rotation_info}{fov_info}{mode_suffix}.mp4"
        video_filepath = os.path.join(self.save_dir, video_filename)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        frames = []
        temp_dir = tempfile.mkdtemp()

        try:
            print(f"  生成视频帧...")
            for layer_idx in tqdm(range(data.shape[2]), desc=f"  Video"):
                frame_fig = self.generate_frame(data, layer_idx, params, slice_thickness, probe_dx)

                temp_frame_path = os.path.join(temp_dir, f"frame_{layer_idx:04d}.png")
                frame_fig.savefig(temp_frame_path, dpi=150, bbox_inches='tight',
                                facecolor='white', pad_inches=0.1)
                frames.append(temp_frame_path)
                plt.close(frame_fig)

            self._encode_video(frames, video_filepath)
            print(f"  视频已保存: {video_filepath}")

        except Exception as e:
            print(f"  生成视频失败: {e}")

        finally:
            for frame_path in frames:
                try:
                    os.remove(frame_path)
                except OSError:
                    pass
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

    def _encode_video(self, frames, video_filepath):
        """编码视频文件"""
        if not frames:
            return

        first_frame = imageio.imread(frames[0])
        h, w = first_frame.shape[:2]

        h_crop = h - (h % 2)
        w_crop = w - (w % 2)

        try:
            with imageio.get_writer(video_filepath, fps=5, codec='libx264',
                                  quality=8, macro_block_size=1) as writer:
                for frame_path in frames:
                    image = imageio.imread(frame_path)
                    if image.shape[0] != h_crop or image.shape[1] != w_crop:
                        image = image[:h_crop, :w_crop]
                    writer.append_data(image)

        except Exception as e:
            print(f"    MP4编码失败: {e}")


def batch_save_videos_and_mat(all_pt_files, current_params, base_save_dir):
    """批量生成视频和MAT文件"""
    print("开始批量生成视频和MAT文件...")

    if not all_pt_files:
        print("未找到任何模型文件进行批量处理")
        return

    print(f"找到 {len(all_pt_files)} 个模型文件需要处理")

    for pt_file_path, region_number in all_pt_files:
        try:
            print(f"\n处理文件: {os.path.basename(pt_file_path)} (区域: {region_number})")
            region_save_dir = os.path.join(base_save_dir, region_number)
            if not os.path.exists(region_save_dir):
                os.makedirs(region_save_dir)

            # 加载数据
            model_data = DataProcessor.load_model_file(pt_file_path)
            file_optimizable_tensors = DataProcessor.extract_optimizable_tensors(model_data)

            # 保存MAT文件
            pt_filename = os.path.basename(pt_file_path)
            ParameterManager.auto_save_mat_file(region_save_dir, file_optimizable_tensors, pt_filename)

            # 检查视频是否已存在
            if check_if_processed(region_save_dir, 'video'):
                print(f"  区域 {region_number} 已存在视频文件，跳过视频生成...")
                continue

            # 生成视频
            file_tensor_np = file_optimizable_tensors['objp'].detach().cpu().numpy()
            file_data = np.transpose(file_tensor_np, (2, 3, 1, 0)).mean(axis=-1)

            region_video_generator = VideoGenerator(region_save_dir)
            file_pt_dir = os.path.dirname(os.path.abspath(pt_file_path))
            region_video_generator.save_video(file_data, current_params, file_pt_dir,
                                     file_optimizable_tensors)

        except Exception as e:
            print(f"处理文件 {os.path.basename(pt_file_path)} 时出错: {e}")

    print("\n批量视频和MAT文件生成完成!")
