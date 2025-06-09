"""
文件操作工具模块
"""

import os
import glob


def create_save_directory(folder_path, region_number):
    """创建保存目录结构"""
    parent_dir = os.path.dirname(folder_path)
    save_dir = os.path.join(parent_dir, 'Data_Saved', region_number)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def check_if_processed(save_dir, check_type='general', force=False):
    """检查是否已经处理过"""
    # 如果设置了强制重新处理，直接返回False
    if force:
        return False
        
    if check_type == 'video':
        # 对于视频生成，检查是否存在mp4文件
        video_files = glob.glob(os.path.join(save_dir, '*.mp4'))
        return len(video_files) > 0
    else:
        # 对于其他情况，检查是否存在png文件
        image_files = glob.glob(os.path.join(save_dir, '*.png'))
        return len(image_files) > 0


def find_pt_files(folder_path, filename_pattern):
    """查找匹配模式的pt文件"""
    found_files = []
    
    if not os.path.exists(folder_path):
        print(f"未找到目录: {folder_path}")
        return found_files
    
    for item in os.listdir(folder_path):
        region_path = os.path.join(folder_path, item)
        if os.path.isdir(region_path):
            for subitem in os.listdir(region_path):
                subdir_path = os.path.join(region_path, subitem)
                if os.path.isdir(subdir_path):
                    pt_file_path = os.path.join(subdir_path, filename_pattern)
                    if os.path.exists(pt_file_path):
                        found_files.append((pt_file_path, item))
                        print(f"找到文件: {pt_file_path} (区域: {item})")
    
    return found_files
