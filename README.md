# PtyRAD数据可视化工具

用于批量处理和可视化PtyRAD重构数据的交互式工具。

## 安装

```bash
cd /path/to/ptyrad_utils
pip install -e .
```

## 使用方法

安装后，可以直接使用 `plot_ptyrad` 命令：

```bash
# 基本用法
plot_ptyrad --folder /path/to/All_Data_PtyRAD --file model_iter1000.pt

# 强制重新处理
plot_ptyrad --folder /path/to/All_Data_PtyRAD --file model_iter1000.pt --force

# 显示详细信息
plot_ptyrad --folder /path/to/All_Data_PtyRAD --file model_iter1000.pt --detailed

# 使用短参数名
plot_ptyrad -f /path/to/data -F model_iter1000.pt --force
```

## 功能特性

- 交互式3D数据可视化
- 批量生成重构数据的视频
- MAT文件自动导出

## 目录结构

```
ptyrad_utils/
├── plot_pt_file.py      # 核心处理模块
├── cli.py               # 命令行接口
├── config.py            # 配置参数
├── ui_components.py     # UI组件
├── data_processor.py    # 数据处理
├── setup.py             # 安装脚本
└── README.md            # 说明文档
```
