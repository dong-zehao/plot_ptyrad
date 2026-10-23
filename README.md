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

核心实现统一位于 `plot_ptyrad/` 包内。命令行使用 `plot_ptyrad` 或 `python -m plot_ptyrad`；脚本导入使用 `from plot_ptyrad.data_processor import DataProcessor` 等包路径。根目录旧模块和 `setup.py` 已移除，安装统一使用 `pyproject.toml`。

```text
plot_ptyrad/
├── pyproject.toml          # 项目元信息、依赖、命令入口与测试配置
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
└── tests/                  # MAT 内容、变换与 CLI 回归测试
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

## Qt 高响应界面

`UI_update` 分支使用 **PySide6（Qt 6）原生控件 + PyQtGraph**。交互画布不再由 Matplotlib 重绘；Matplotlib 仅用于生成原有格式的高分辨率 PNG 和视频帧。

### 安装与启动

需要 Python **3.9 或更新版本**；建议使用 Python 3.10–3.12。更新代码后，在运行 PtyRAD 的环境中重新安装依赖：

```bash
pip install -e ".[dev]"
plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.pt --force
```

新增依赖是 `PySide6-Essentials>=6.5,<7` 和 `pyqtgraph>=0.13.7,<0.14`。Essentials 提供本项目所需的 Qt 控件和绘图模块，无需安装 WebEngine。界面不强制依赖 OpenGL 或独立显卡。CLI 参数、文件发现规则和 `Data_Saved/<region>/` 输出位置保持一致。

### 布局与操作

- 左上为图像和色条；右侧依次是保存按钮、Display Mode、Color Schemes 和 FFT Contrast；底部七组滑块仍按 Layer Start、Layer Count、Rotation、Crop X/Y、Center X/Y 排列；右下保留 Next Region 和 End。
- 数字输入框按 Enter 或离开焦点时应用，滑块与输入框双向同步；旋转精度为 0.1°，FFT Gamma 精度为 0.01。
- 保留五种原有色图、原图/FFT 切换、连续层叠加、旋转、裁剪与中心偏移。Crop X 仍对应第一空间维，Crop Y 对应第二空间维。
- 图像支持鼠标拖动平移、滚轮缩放；右键菜单中的 View All 或缩放后出现的 A 按钮可恢复完整视图。
- 参数修改后延迟约 350 ms 自动保存，切换区域、结束或关闭窗口时立即保存。标题栏关闭窗口仍进入下一区域；End 结束整个处理流程。
- Save Image 和 Save All Videos & MAT 使用**点击时的参数快照**，在后台导出。导出时仍可调整预览；保存和切换区域按钮暂时禁用，完成后恢复，防止重复导出或中途关闭。失败信息显示在右侧状态区及终端。
- PNG 继续使用 600 dpi 输出；视频与 MAT 的内容、命名和已有视频的跳过规则保持一致。预览降采样只作用于屏幕显示，导出仍使用完整分辨率数据。

### 响应速度优化

预览在独立工作线程执行；主线程只处理控件与显示。每轮最多保留一个正在运行的请求和一个最新请求，约每 16 ms 检查结果，丢弃过时画面，避免快速拖动后逐帧追赶。

缓存仅保留最近一次的二维中间结果，不建立第二份完整三维体：切换色图只更新查找表；改变 Gamma 复用 FFT 功率；改变裁剪复用已旋转的图像；相同层范围复用叠加结果。旋转仍使用原来的 SciPy 插值，FFT 仍使用原来的高斯窗和数值公式。

### 开发与验证

```bash
# Qt 在无桌面环境中运行真实控件及事件循环
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

测试包含数值对照、控件同步、快速拖动合并、后台导出与异常恢复、区域切换、单像素图像，以及真实 PNG/MP4/MAT 输出。无窗口 Qt 测试通过 `create_interface(block=False)` 嵌入事件循环；正常 CLI 仍按区域阻塞，等待用户操作。`MPLBACKEND=Agg` 仅影响 Matplotlib，不能用于关闭 Qt 界面。

新增模块：`render_engine.py` 管理分阶段预览缓存，`image_export.py` 提供独立 Agg 图片导出；`interactive_plotter.py` 负责后台任务和界面事件，`ui_components.py` 定义原生 Qt 布局与控件。
