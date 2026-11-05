"""
视频生成模块
"""

import os
import tempfile
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import imageio.v2 as imageio
from tqdm import tqdm

from .config import MATPLOTLIB_CONFIG
from .data_processor import DataProcessor
from .file_utils import check_if_processed
from .mat_export import save_model_mat_for_video


class VideoGenerator:
    """视频生成工具类"""

    def __init__(self, save_dir):
        self.save_dir = save_dir

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
        frame_fig = Figure(figsize=(10, 8))
        FigureCanvasAgg(frame_fig)
        frame_ax = frame_fig.subplots()
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

        frame_cb = frame_fig.colorbar(frame_im, ax=frame_ax, shrink=0.8)
        frame_cb.ax.tick_params(labelsize=MATPLOTLIB_CONFIG['ytick.labelsize'])
        frame_cb.set_label(cbar_label, fontsize=12, color='#2c3e50')

        return frame_fig

    def save_video(self, data, params, pt_file_dir, optimizable_tensors=None, report=None):
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
            slice_thickness = DataProcessor._to_numpy(optimizable_tensors['slice_thickness']).item()

        video_filename = f"objp_layers_video{rotation_info}{fov_info}{mode_suffix}.mp4"
        video_filepath = os.path.join(self.save_dir, video_filename)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        frames = []
        temp_dir = tempfile.mkdtemp()

        try:
            print(f"  生成视频帧...")
            if report:
                report(f"生成 {video_filename}", 0, data.shape[2])
            for layer_idx in tqdm(range(data.shape[2]), desc=f"  Video"):
                frame_fig = self.generate_frame(data, layer_idx, params, slice_thickness, probe_dx)

                temp_frame_path = os.path.join(temp_dir, f"frame_{layer_idx:04d}.png")
                frame_fig.savefig(temp_frame_path, dpi=150, bbox_inches='tight',
                                facecolor='white', pad_inches=0.1)
                frames.append(temp_frame_path)
                frame_fig.clear()
                if report:
                    report(f"生成 {video_filename}", layer_idx + 1, data.shape[2])

            self._encode_video(frames, video_filepath, report)
            print(f"  视频已保存: {video_filepath}")

        except Exception as e:
            print(f"  生成视频失败: {e}")
            raise

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

    def _encode_video(self, frames, video_filepath, report=None):
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
                stage = f"编码 {os.path.basename(video_filepath)}"
                if report:
                    report(stage, 0, len(frames))
                for index, frame_path in enumerate(frames, 1):
                    image = imageio.imread(frame_path)
                    if image.shape[0] != h_crop or image.shape[1] != w_crop:
                        image = image[:h_crop, :w_crop]
                    writer.append_data(image)
                    if report:
                        report(stage, index, len(frames))

        except Exception as e:
            print(f"    MP4编码失败: {e}")
            raise


def batch_save_videos_and_mat(all_pt_files, current_params, base_save_dir, progress=None):
    """批量生成视频和MAT文件，返回失败项供界面报告。"""
    failures = []
    print("开始批量生成视频和MAT文件...")

    if not all_pt_files:
        print("未找到任何模型文件进行批量处理")
        return failures

    print(f"找到 {len(all_pt_files)} 个模型文件需要处理")

    for index, (pt_file_path, region_number) in enumerate(all_pt_files, 1):
        def report(stage, done=0, total=0):
            if progress:
                progress(f"{index}/{len(all_pt_files)} · {region_number} · {stage}",
                         os.path.abspath(pt_file_path), done, total)

        try:
            print(f"\n处理文件: {os.path.basename(pt_file_path)} (区域: {region_number})")
            region_save_dir = os.path.join(base_save_dir, region_number)
            if not os.path.exists(region_save_dir):
                os.makedirs(region_save_dir)

            mat_name = os.path.splitext(os.path.basename(pt_file_path))[0] + '.mat'
            has_mat = check_if_processed(region_save_dir, 'mat', filename=mat_name)
            has_video = check_if_processed(region_save_dir, 'video')
            if has_mat:
                message = f"已有 {mat_name}，跳过 MAT 保存"
                print(f"  {message}")
                report(message, 1, 1)
            else:
                file_optimizable_tensors = save_model_mat_for_video(
                    pt_file_path, region_save_dir, current_params, report)

            if has_video:
                print(f"  区域 {region_number} 已存在视频文件，跳过视频生成...")
                report("已有视频，跳过生成", 1, 1)
                continue

            if has_mat:
                report("读取视频所需的模型数据")
                model = DataProcessor.load_model_file(pt_file_path)
                file_optimizable_tensors = DataProcessor.extract_optimizable_tensors(model)
            file_data = DataProcessor.prepare_object_volume(file_optimizable_tensors['objp'])
            region_video_generator = VideoGenerator(region_save_dir)
            file_pt_dir = os.path.dirname(os.path.abspath(pt_file_path))
            region_video_generator.save_video(file_data, current_params, file_pt_dir,
                                              file_optimizable_tensors, report)

        except Exception as e:
            print(f"处理文件 {os.path.basename(pt_file_path)} 时出错: {e}")
            failures.append(f"{pt_file_path}: {e}")

    print("\n批量视频和MAT文件生成完成!")
    return failures
