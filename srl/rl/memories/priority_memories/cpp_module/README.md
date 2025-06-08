# building pybind11-based C++ extension

1. Please install a C compiler such as GCC.
2. Please install pybind11.(`pip install pybind11`)
3. Execute the following and build it.
``` bash
> python setup_cpp.py build_ext
```


# Directory structure

```
cpp_module/
├─ bin/          # ビルド成果物格納先
├─ build/        # 一時ビルドファイル（毎回削除される）
├─ src/          # C++ソースコード
└─ setup_cpp.py  # ビルド用
```


# history
2025/04/30: ProportionalMemory.cppを追加（pyの実装より10倍以上高速化）
