# Qt GUI 版本 PtyRAD数据可视化工具

该版本提供基于 Qt 的图形界面后端，用于交互式查看和处理 PtyRAD 重构结果。

## 新用户环境安装

本项目建议在独立的 Conda 环境中使用，避免和其他 Python 项目的依赖发生冲突。

## 安装项目和依赖

首次使用，或 `setup.py` 依赖发生变化后，运行：

```powershell
conda activate ptyrad
pip install -e .
```

该命令会安装 `plot_ptyrad` 命令，并安装主要依赖，包括 `torch`、`h5py`、`matplotlib`、`scipy`、`PySide6` 和 `pyqtgraph`。

## 检查安装是否成功

```powershell
plot_ptyrad --help
```

如果安装正确，应能看到以下参数：

```text
--gui-backend {matplotlib,qt}
--preview-max-size PREVIEW_MAX_SIZE
```

### 1. 激活运行环境

打开 PowerShell，并激活 `ptyrad` conda 环境：

```powershell
conda activate ptyrad
```

### 2. 进入项目目录

```powershell
cd /path/to/plot_ptyrad
```

### 3. 确认当前分支

确认当前位于 `qt-gui` 分支：

```powershell
git branch --show-current
```

期望输出为：

```text
qt-gui
```

### 4. 启动 Qt GUI

使用以下命令启动 Qt GUI：

```powershell
plot_ptyrad --folder "/path/to/parent_folder" --file model_iter1000.hdf5 --gui-backend qt --force
```

其中：

- `--folder` 指定包含所有重构 region 的父文件夹。
- `--file` 指定需要搜索和读取的重构结果文件名。
- `--gui-backend qt` 表示使用 Qt GUI 后端。
- `--force` 表示即使某个 region 已经处理过，也强制重新打开 GUI。

### 5. 不强制重新打开已处理 region

如果不想强制重新打开已经处理过的 region，可以去掉 `--force`：

```powershell
plot_ptyrad --folder "/path/to/parent_folder" --file model_iter1000.hdf5 --gui-backend qt
```

### 6. 如果 GUI 预览卡顿，降低预览尺寸

如果交互预览比较卡顿，可以降低预览最大尺寸：

```powershell
plot_ptyrad --folder "/path/to/parent_folder" --file model_iter1000.hdf5 --gui-backend qt --force --preview-max-size 768
```

`--preview-max-size` 用于控制 GUI 预览图像的最大边长。数值越小，交互越流畅，但预览分辨率也会降低。
如果希望预览更清晰，可以提高预览尺寸：

```powershell
plot_ptyrad --folder "F:\Data_ARM300\20260510 SCO_96K\All_Data" --file model_iter1000.hdf5 --gui-backend qt --force --preview-max-size 1536
```

保存 PNG、视频和 MAT 文件时仍使用原始 full-resolution 数据，不受预览降采样影响。

### 推荐常用命令

一般情况下，推荐使用以下流程：

```powershell
conda activate ptyrad
cd /path/to/plot_ptyrad
plot_ptyrad --folder "/path/to/parent_folder" --file model_iter1000.hdf5 --gui-backend qt --force
```
