# PtyRAD数据可视化工具

用于批量处理和可视化PtyRAD重构数据的交互式工具。

## 功能特性

- 交互式3D数据可视化
- 批量生成重构数据的视频
- MAT文件自动导出

## 安装
将 plot_ptyrad 安装在`/path/plot_ptyrad/`路径的步骤如下：
```bash
conda activate ptyrad

cd /path/
git clone https://github.com/dong-zehao/plot_ptyrad.git

cd ./plot_ptyrad
pip install -e .
```

## 使用方法

安装后，可以直接使用 `plot_ptyrad` 命令：

```bash
# 基本用法: 处理parent_folder文件夹下的重构输出，支持 .pt 或 .hdf5
# 数据文件夹组织形式为  ./parent_folder/region_name/some/nested/folders/model_iter1000.pt
# 输出文件的结构:      ./parent_folder/Data_Saved/region_name/saved_file.png
plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.pt

# 同样支持 hdf5 文件
plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.hdf5

# 若不想跳过已经处理过的数据，可强制重新处理
plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.pt --force

# 使用短参数名
plot_ptyrad -f /path/to/parent_folder -F model_iter1000.pt --force
```

## 数据文件夹目录结构

```
/path/to/parent_folder/
├── 4Dregion01/
│   └── some/nested/folder/
│       └── model_iter1000.pt          # 深层嵌套（也可为 .hdf5）
├── 4Dregion02/
│   └── any_structure/
│       └── sub/folder/
│           └── model_iter1000.pt      # 任意结构（也可为 .hdf5）
├── ...
│
└── Data_Saved/
    ├── plot_params.json          # 全局参数文件
    ├── 4Dregion01/               # 区域1的处理结果
    │   ├── *.png
    │   ├── *.mp4
    │   └── *.mat
    ├── 4Dregion02/               # 区域2的处理结果
        ├── *.png
        ├── *.mp4
        └── *.mat
```

## 项目结构与开发

核心实现位于独立的 Python 包内，根目录同名模块仅用于兼容已有脚本和 notebook 的导入。

```text
plot_ptyrad/
├── pyproject.toml          # 项目元信息、依赖、命令入口与测试配置
├── setup.py                # 传统安装入口
├── cli.py                  # 兼容 python cli.py
├── plot_ptyrad/
│   ├── __init__.py
│   ├── __main__.py         # python -m plot_ptyrad
│   ├── cli.py              # 命令行参数与处理循环
│   ├── plot_pt_file.py     # 单个区域的处理流程
│   ├── data_processor.py   # 模型加载、变换与参数管理
│   ├── mat_export.py       # 完整张量和 object_3d 导出
│   ├── interactive_plotter.py
│   ├── ui_components.py
│   ├── video_generator.py
│   ├── file_utils.py
│   └── config.py
├── notebooks/              # 探索用 notebook
└── tests/                  # MAT 内容、变换、兼容入口与 CLI 回归测试
```

```bash
pip install -e ".[dev]"
python -m pytest
python -m plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.hdf5
```

## MAT 导出内容

点击 **Save All Videos & MAT** 时，每个模型对应一个同名 MAT 文件：

- 对 HDF5 递归读取所有组中的数值数据集，包括整数、浮点数、布尔和复数张量以及数值标量；不再限于 `optimizable_tensors`。
- 原始张量保持原始数值和维度（MATLAB 的标量和一维数组以二维形式表示）。`optimizable_tensors` 中的变量优先保留旧名称；其他变量用完整组路径生成合法 MATLAB 名称。非法字符改为下划线，名称以字母开头且不超过 63 字符，同名时添加数字后缀。
- `tensor_paths` 是两列 cell 数组，分别记录 MAT 变量名与 HDF5 数据集路径，可用于准确定位原始张量。
- `object_3d` 的维度为 **(height, width, depth)**：将原始 `objp(mode, depth, height, width)` 沿 mode 取平均，再逐层应用当前 `plot_params` 的旋转、裁剪与裁剪中心偏移。保留全部深度层，不受 `start_layer`、`layer_count`、色图或 FFT 显示模式影响。裁剪坐标约定与现有界面一致：`crop_x` 对应第一空间维，`crop_y` 对应第二空间维。
- HDF5 自带的 `object_3d` 不会被丢弃，会另分配不冲突的名称并记录在 `tensor_paths`。
- 字符串、对象引用、复合记录和 null 数据集不是数值张量，会列入 `skipped_datasets` 并输出提示；HDF5 attributes 不作为张量导出。
- 再次点击会更新对应模型的 MAT，以便保存最新的旋转裁剪参数；已有视频的跳过逻辑保持不变。写入失败时保留原 MAT。
- `.pt` 仍导出 `optimizable_tensors` 中的数值张量，并同样新增 `object_3d`。

采用 SciPy MAT v5 格式，单个变量受该格式约 2 GB 的大小限制，超限将报告保存失败。

### 已验证环境

回归测试使用 Python 3.12、PyTorch 2.2.2、NumPy 1.26.4，采用无窗口模式生成图像和视频。若使用旧版 PyTorch 2.2.2，需配合 NumPy 1.x；NumPy 2.x 会导致张量与 NumPy 数组转换失败。建议在独立环境安装兼容版本，例如 `pip install "numpy<2"`。
