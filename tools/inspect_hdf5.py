import h5py
import numpy as np
import sys

def print_hdf5_info(file_path, max_data_print=10):
    """
    格式化遍历并打印HDF5文件完整结构
    :param file_path: hdf5文件路径
    :param max_data_print: 数组最多打印前N个元素
    """
    def recursive_print(obj, indent=0):
        indent_str = "  " * indent
        if isinstance(obj, h5py.Dataset):
            print(f"{indent_str}[DATASET] {obj.name}")
            print(f"{indent_str}  dtype: {obj.dtype}")
            print(f"{indent_str}  shape: {obj.shape}")
            print(f"{indent_str}  size: {obj.size:,}")
            print(f"{indent_str}  chunks: {obj.chunks}")
            print(f"{indent_str}  compression: {obj.compression}")

            if obj.size == 0:
                print(f"{indent_str}  data: (empty)")
            elif obj.size <= max_data_print:
                print(f"{indent_str}  data: {obj[()]}")
            else:
                arr = obj[()]
                flat = arr.flatten()[:max_data_print]
                print(f"{indent_str}  data preview (first {max_data_print}): {flat} ...")

            attrs = dict(obj.attrs)
            if attrs:
                print(f"{indent_str}  attributes:")
                for k, v in attrs.items():
                    print(f"{indent_str}    {k}: {v}")
            print()

        elif isinstance(obj, h5py.Group):
            print(f"{indent_str}[GROUP] {obj.name}")
            attrs = dict(obj.attrs)
            if attrs:
                print(f"{indent_str}  attributes:")
                for k, v in attrs.items():
                    print(f"{indent_str}    {k}: {v}")
            print()
            for key in obj:
                recursive_print(obj[key], indent + 1)

    with h5py.File(file_path, "r") as f:
        print("=" * 60)
        print(f"HDF5 File Path: {file_path}")
        print(f"File Driver: {f.driver}")
        print("=" * 60)
        recursive_print(f)
    print("\n===== 解析完成 =====")


if __name__ == "__main__":
    # 命令行参数判断
    if len(sys.argv) < 2:
        print("使用方法：")
        print(f"  python {sys.argv[0]} your_file.h5")
        print("示例：")
        print(f"  python {sys.argv[0]} data/test.hdf5")
        sys.exit(1)

    hdf5_file = sys.argv[1]
    try:
        print_hdf5_info(hdf5_file, max_data_print=15)
    except FileNotFoundError:
        print(f"错误：文件「{hdf5_file}」不存在！")
    except Exception as e:
        print(f"读取HDF5失败：{type(e).__name__}: {e}")
