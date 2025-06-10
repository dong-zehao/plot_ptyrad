#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py - 安装脚本

用于安装plot_ptyrad命令行工具
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "PtyRAD数据可视化工具"

setup(
    name="plot-ptyrad",
    description="PtyRAD重构数据可视化工具",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="dong-zehao",
    url="https://github.com/dong-zehao/plot_ptyrad",
    packages=find_packages(),
    py_modules=[
        'plot_pt_file',
        'config', 
        'ui_components',
        'data_processor',
        'interactive_plotter',
        'video_generator',
        'file_utils',
        'cli'
    ],
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'scipy>=1.7.0',
        'PyYAML>=6.0',
        'imageio>=2.19.0',
        'imageio-ffmpeg>=0.4.7',  # 添加 ffmpeg 支持
        'tqdm>=4.62.0',
    ],
    entry_points={
        'console_scripts': [
            'plot_ptyrad=cli:main',
        ],
    },
    python_requires=">=3.8",
)
