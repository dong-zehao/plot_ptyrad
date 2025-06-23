"""
文件操作工具模块
"""

import os
import glob


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
    
    # 第一层：遍历FOLDER下的所有项目（作为区域名）
    for item in sorted(os.listdir(folder_path)):
        region_path = os.path.join(folder_path, item)
        if os.path.isdir(region_path): 
            region_name = item  # 第一层子目录作为区域名
            
            # 递归搜索该区域目录下的所有.pt文件
            for root, dirs, files in os.walk(region_path):
                for file in files:
                    if file == filename_pattern:  # 精确匹配文件名
                        pt_file_path = os.path.join(root, file)
                        found_files.append((pt_file_path, region_name))
                        print(f"找到文件: {pt_file_path} (区域: {region_name})")
    
    return found_files
