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
