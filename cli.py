#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli.py - PtyRAD数据可视化工具的命令行接口

提供plot_ptyrad命令来批量处理和可视化PtyRAD重构数据
"""

import argparse
import sys
import os

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog='plot_ptyrad',
        description="""
plot_ptyrad 用于批量解析 ptyrad 重构的 .pt/.hdf5 文件并生成交互式可视化. 预期的文件夹组织结构为 /FOLDER/region_name/some/nested/folders/FILE. 
其中 FOLDER 和 FILE 是用户指定的参数, region_name 作为区域编号.
保存的图片和视频文件在/FOLDER/Data_Saved/region_name/ 目录下, 通过检测该目录中是否有视频和图片文件来判断是否已经处理过数据, 如果已经处理则跳过这组数据.
如果指定了 --force 参数, 则会强制重新处理所有数据, 不管之前是否已经处理过.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.pt
    plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.hdf5
  plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.pt --force
  plot_ptyrad -f /path/to/parent_folder -F model_iter1000.pt --force
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--folder', '-f',
        type=str,
        required=True,
        help='存有多组ptyrad重构结果的文件夹路径.',
    )
    
    parser.add_argument(
        '--file', '-F',
        type=str,
        required=True,
        help='模型文件名 (例如: model_iter1000.pt 或 model_iter1000.hdf5)'
    )
    
    # 可选参数
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新处理所有数据，不跳过已处理的文件'
    )
    
    return parser

def validate_args(args):
    """验证命令行参数"""
    # 检查文件夹是否存在
    if not os.path.exists(args.folder):
        print(f"错误: 文件夹不存在: {args.folder}")
        return False
    
    if not os.path.isdir(args.folder):
        print(f"错误: 指定路径不是目录: {args.folder}")
        return False
    
    # 检查文件名格式
    if not args.file.endswith(('.pt', '.hdf5', '.h5')):
        print(f"警告: 文件名 '{args.file}' 既不以 .pt 也不以 .hdf5/.h5 结尾")
    
    return True

def main():
    """CLI主入口函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 验证参数
    if not validate_args(args):
        sys.exit(1)
    
    # 导入核心处理模块
    try:
        from plot_pt_file import (
            find_pt_files, 
            process_single_file, 
            DataProcessor,
            PROCESSING_STATE
        )
    except ImportError as e:
        print(f"错误: 无法导入必要模块: {e}")
        print("请确保所有依赖文件都在正确位置")
        sys.exit(1)
    
    try:
        # 查找pt文件
        pt_files = find_pt_files(args.folder, args.file)
        
        if not pt_files:
            print(f"未找到匹配的文件: {args.file}")
            sys.exit(1)
        
        print(f"找到 {len(pt_files)} 个文件需要处理")
        
        if args.force:
            print("强制重新处理模式：将重新处理所有数据文件")
        
        # 处理文件
        successful_count = 0
        
        for pt_file_path, region_number in pt_files:
            # 检查是否需要结束处理
            if PROCESSING_STATE['end_processing']:
                print("用户选择结束处理，退出循环")
                break
            
            # 处理文件
            if process_single_file(pt_file_path, region_number, args.folder, pt_files, force=args.force):
                successful_count += 1
                
                # 检查是否需要结束处理
                if PROCESSING_STATE['end_processing']:
                    print("用户选择结束处理，退出循环")
                    break
            else:
                # 如果处理失败且是因为用户选择结束，则退出循环
                if PROCESSING_STATE['end_processing']:
                    break
        
        print(f"\n处理完成! 成功处理了 {successful_count}/{len(pt_files)} 个文件。")
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
