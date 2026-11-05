"""完整 HDF5 数值数据集和处理后三维物体的 MAT 导出。"""

import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np
from scipy.io import savemat

from .data_processor import DataProcessor


def _mat_name(path, used):
    # 保留旧 optimizable_tensors 的顶层变量名，其余以组路径命名。
    prefix = "optimizable_tensors/"
    name = path[len(prefix):] if path.startswith(prefix) else path
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not name or not name[0].isalpha():
        name = "tensor_" + name
    name = name[:63]
    candidate = name
    index = 2
    while candidate in used:
        suffix = "_" + str(index)
        candidate = name[:63 - len(suffix)] + suffix
        index += 1
    used.add(candidate)
    return candidate


def _collect_tensors(optimizable_tensors, source_path):
    tensors = {}
    skipped = []
    if source_path and Path(source_path).suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(source_path, "r") as source:
            def collect(path, node):
                if not isinstance(node, h5py.Dataset):
                    return
                # 字符串、对象引用、复合记录和无数据的 null dataset 不是数值张量。
                if node.shape is None or node.dtype.kind not in "biufc":
                    skipped.append(path)
                    return
                tensors[path] = node[()]
            source.visititems(collect)
            # visititems 只访问硬链接的首个路径，并会跳过软链接。
            for path in ("optimizable_tensors/objp", "optimizable_tensors/slice_thickness",
                         "slice_thickness"):
                if path not in tensors and path not in skipped and path in source:
                    collect(path, source.get(path))
    else:
        for name, value in optimizable_tensors.items():
            array = DataProcessor._to_numpy(value)
            if array.dtype.kind in "biufc":
                tensors["optimizable_tensors/" + name] = array
            else:
                skipped.append(name)
    return tensors, skipped


def save_mat_file(save_dir, optimizable_tensors, filename=None, plot_params=None,
                  source_path=None):
    """将所有数值张量和 object_3d 原子写入同一个 MAT v5 文件。

    tensor_paths 为两列 cell 数组：MAT 变量名、原始数据集路径。
    错误向调用者传播；失败时保留已有 MAT，不留下半写入文件。
    """
    if optimizable_tensors is None or "objp" not in optimizable_tensors:
        raise ValueError("导出 object_3d 需要 optimizable_tensors/objp")
    params = plot_params or {}
    tensors, skipped = _collect_tensors(optimizable_tensors, source_path)
    return _write_mat(save_dir, filename, params, tensors, skipped, optimizable_tensors["objp"])


def save_model_mat_for_video(source_path, save_dir, params, report):
    """导出一个模型，并返回视频所需的原始数据，避免重复读取 HDF5。"""
    filename = Path(source_path).name
    report("读取 " + filename)
    if Path(source_path).suffix.lower() in (".h5", ".hdf5"):
        tensors, skipped = _collect_tensors(None, source_path)
        objp = tensors.get("optimizable_tensors/objp")
        if objp is None:
            raise ValueError("导出 object_3d 需要 optimizable_tensors/objp")
        video_inputs = {"objp": objp}
        thickness = tensors.get("optimizable_tensors/slice_thickness", tensors.get("slice_thickness"))
        if thickness is not None:
            video_inputs["slice_thickness"] = thickness
    else:
        model = DataProcessor.load_model_file(source_path)
        video_inputs = DataProcessor.extract_optimizable_tensors(model)
        tensors, skipped = _collect_tensors(video_inputs, None)
    _write_mat(save_dir, filename, params, tensors, skipped, video_inputs["objp"], report)
    return video_inputs


def _write_mat(save_dir, filename, params, tensors, skipped, objp, report=None):
    used = {"object_3d", "tensor_paths", "skipped_datasets"}
    mat_data = {}
    paths = []
    # 优先为旧张量分配名称，避免根目录的同名数据覆盖它们。
    for path in sorted(tensors, key=lambda p: (not p.startswith("optimizable_tensors/"), p)):
        name = _mat_name(path, used)
        mat_data[name] = tensors[path]
        paths.append((name, path))
    if report:
        report("计算 object_3d")
    mat_data["object_3d"] = DataProcessor.transform_object_volume(objp, params)
    mat_data["tensor_paths"] = np.asarray(paths, dtype=object).reshape(-1, 2)
    mat_data["skipped_datasets"] = np.asarray(skipped, dtype=object)
    os.makedirs(save_dir, exist_ok=True)
    target = Path(save_dir) / ((Path(filename).stem if filename else "model") + ".mat")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=save_dir, suffix=".mat", delete=False) as temporary:
            temporary_path = temporary.name
            # 同一文件流中依次写入变量，SciPy 只在流开头写 MAT v5 文件头。
            for index, (name, value) in enumerate(mat_data.items()):
                if report:
                    report(f"写入 {target.name} · {name}（按变量计）", index, len(mat_data))
                savemat(temporary.file, {name: value}, format="5", do_compression=False,
                        long_field_names=True)
            if report:
                report(f"写入 {target.name} · 完成", len(mat_data), len(mat_data))
        os.replace(temporary_path, target)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    if skipped:
        print("  跳过非数值或空数据集: " + ", ".join(skipped))
    print(f"MAT文件已保存到: {target}")
    return str(target)
