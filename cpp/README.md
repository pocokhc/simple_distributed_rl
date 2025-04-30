# building pybind11-based C++ extension

1. Please install a C compiler such as GCC.
2. Please install pybind11.(`pip install pybind11`)
3. Execute the following and build it.
``` bash
> cd cpp
> python setup_cpp.py build_ext
```

4. There is an executable file under `bin`, so please copy it to `srl/rl/memories/priority_memories/bin`.


# Directory structure

```
cpp/
├── bin/        # .so 実行モジュール出力先(OS別に配置)
│   ├── macos/
│   ├── linux/
│   └── win32/
│
├── build/      # ビルド中間ファイル(都度削除されます)
│
├── src/
│   └── proportional_memory.cpp
│
└── setup_cpp.py   # ビルド用
```


# history
2025/04/30: ProportionalMemory.cppを追加（pyの実装より12倍以上高速化）

