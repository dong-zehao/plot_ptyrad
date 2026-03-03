#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pt_file.py - 重构版本

PtyRAD重构数据可视化工具主模块
"""

import os
import matplotlib.pyplot as plt

# 导入自定义模块
from config import PROCESSING_STATE, MATPLOTLIB_CONFIG
from data_processor import DataProcessor, ParameterManager
from interactive_plotter import InteractivePlotter
from file_utils import check_if_processed, find_model_files


def plot_tensor_overview(optimizable_tensors, pt_file_dir, save_dir, all_pt_files=None, pt_filename=None):
    """创建张量概览图"""
    # 输出关键参数
    if 'obj_tilts' in optimizable_tensors:
        obj_tilts = optimizable_tensors['obj_tilts'].detach().cpu().numpy()
        print(f"\nobj_tilts 数值: {obj_tilts}")
    
    if 'slice_thickness' in optimizable_tensors:
        slice_thickness = optimizable_tensors['slice_thickness'].detach().cpu().numpy()
        print(f"slice_thickness 数值: {slice_thickness}")
    
    # 自动检查并保存MAT文件
    ParameterManager.auto_save_mat_file(save_dir, optimizable_tensors, pt_filename)
    
    # 绘制objp
    if 'objp' in optimizable_tensors:
        print("\n正在创建objp交互式图像...")
        plotter = InteractivePlotter(optimizable_tensors['objp'], pt_file_dir, save_dir, 
                             optimizable_tensors, all_pt_files)
        plotter.create_interface()
    else:
        print("没有找到objp张量数据")


def process_single_file(pt_file_path, region_number, all_data_folder_path, all_pt_files=None, force=False):
    """处理单个模型文件（.pt/.hdf5）"""
    try:
        save_dir = os.path.join(all_data_folder_path, 'Data_Saved', region_number)
        os.makedirs(save_dir, exist_ok=True)
        
        if check_if_processed(save_dir, 'general', force=force):
            print(f"区域 {region_number} 已经处理过，跳过...")
            return True
        
        print(f"\n处理区域 {region_number}: {pt_file_path}")
        
        data = DataProcessor.load_model_file(pt_file_path)
        optimizable_tensors = DataProcessor.extract_optimizable_tensors(data)
        pt_file_dir = os.path.dirname(os.path.abspath(pt_file_path))
        
        # 获取pt文件名
        pt_filename = os.path.basename(pt_file_path)
        
        plot_tensor_overview(optimizable_tensors, pt_file_dir, save_dir, all_pt_files, pt_filename)
        
        # 检查是否需要结束处理
        global PROCESSING_STATE
        if PROCESSING_STATE['end_processing']:
            print(f"用户选择结束处理")
            return False
        
        print(f"区域 {region_number} 处理完成!")
        return True
        
    except Exception as e:
        print(f"处理区域 {region_number} 时出错: {e}")
        return False


def find_pt_files(folder_path, filename_pattern):
    """向后兼容：保留旧导入名"""
    return find_model_files(folder_path, filename_pattern)