"""
setup_cpp.py for building pybind11-based C++ extension
"""

import glob
import os
import shutil
import sys

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# ---------------------------------------
# コンパイル・リンクオプション（OS別）
# ---------------------------------------
COMPILE_ARGS = {
    # -------------------------------
    #  macOS (Clang, LLVM)
    # -------------------------------
    "darwin": dict(
        extra_compile=[
            "-std=c++17",  # C++17 に準拠
            "-O3",  # 最適化レベル最大（高速化重視）
            "-ffast-math",  # 浮動小数の厳密性を犠牲にして高速化
            "-Xpreprocessor",  # OpenMP 用（clang のために必要）
            "-fopenmp",  # OpenMP 有効化（brew install libomp 推奨）
            # "-march=native",    # 実行中のCPUに最適化(M1,M2などの個別対応）無効とする
        ],
        extra_link=[
            "-lomp"  # OpenMP ランタイムライブラリ（libomp）
        ],
    ),
    # -------------------------------
    #  Linux (GCC / Clang)
    # -------------------------------
    "linux": dict(
        extra_compile=[
            "-std=c++17",  # C++17 に準拠
            "-O3",  # 高速化最適化
            # "-mavx2",             # AVX2 命令を有効にする
            # "-mavx512f",         # AVX512 命令を有効にする (ソースコードを分けて別でコンパイル)
            "-mfma",  # FMA命令を有効にする
            "-DGENERIC_X86",  # x86系CPU向け
            "-ffast-math",  # 浮動小数演算の高速化（IEEE非準拠）
            "-funroll-loops",  # ループ展開（処理を並列・高速化）
            "-ffp-contract=fast",  # FMA（積和）命令使用許可（演算マージ）
            "-funsafe-math-optimizations",  # 浮動小数点の精度を犠牲にした高速化
            "-Wno-attributes",  #
            # "-march=native",      # コード内で分岐最適化のため無効
        ],
        extra_link=[
            "-fopenmp"  # OpenMP をリンク
        ],
    ),
    # -------------------------------
    #  Windows (MSVC)
    # -------------------------------
    "win32": dict(
        extra_compile=[
            "/std:c++17",  # C++17 に準拠
            "/O2",  # 最適化（速度重視）
            "/fp:fast",  # 浮動小数演算の高速化（精度より速度）
            "/Ob2",  # 関数のインライン展開を最大限に有効化
            "/Oi",  # 内蔵関数の使用（例: abs → __builtin_abs）
            "/openmp",  # OpenMP を有効化
            "/wd4244",  # warning 出力抑制
            # "/arch:AVX2",  # AVX2 命令を有効にする
        ],
        extra_link=[
            # 特別なリンクオプションは不要（OpenMP は自動リンクされる）
        ],
    ),
}


# ---------------------------------------
# 現在のプラットフォームに応じた設定を返す
# ---------------------------------------
def get_platform_config():
    system = sys.platform
    if system in COMPILE_ARGS:
        return system, COMPILE_ARGS[system]
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


# ---------------------------------------
# build/ディレクトリのパス
# ---------------------------------------
def get_build_base():
    return os.path.join(os.path.dirname(__file__), "build")


# ---------------------------------------
# build_ext コマンドを拡張して bin/ へ .so/.pyd をコピー
# ---------------------------------------
class CustomBuildExt(build_ext):
    def build_extensions(self):
        super().build_extensions()

        build_dir = os.path.abspath(self.build_lib)
        bin_dir = os.path.join(os.path.dirname(__file__), "bin", sys.platform)
        os.makedirs(bin_dir, exist_ok=True)

        for file in glob.glob(os.path.join(build_dir, f"{self.distribution.get_name()}*.*")):
            shutil.copy(file, bin_dir)
            print(f" Copied {os.path.basename(file)} → {bin_dir}")


# ---------------------------------------
# メイン関数：指定されたモジュール名をビルド
# ---------------------------------------
def build_extension(base_name: str | list[str]):
    if isinstance(base_name, str):
        base_name = [base_name]

    # build/を毎回削除する（クリーンビルド）
    build_dir = get_build_base()
    if os.path.exists(build_dir):
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)

    # platform毎の設定情報を取得
    platform, compile_opts = get_platform_config()

    module_name = f"{base_name[0]}_cpp"  # Python側でimportする名前
    cpp_files = [os.path.join(os.path.dirname(__file__), "src", name + ".cpp") for name in base_name]
    print(f"platform:{platform}, cpp_files:{cpp_files}")

    ext_modules = [
        Extension(
            module_name,
            sources=cpp_files,
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=compile_opts["extra_compile"],
            extra_link_args=compile_opts["extra_link"],
        )
    ]

    setup(
        name=module_name,
        version="0.1",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CustomBuildExt},
        options={"build": {"build_base": build_dir}},
    )


# ---------------------------------------
# 他モジュールをビルドしたいときは
# build_extension(モジュール名)を、変更する(".cpp"は省略)
# 複数ファイルで構成の場合はlistで渡す
# ---------------------------------------
if __name__ == "__main__":
    build_extension("proportional_memory")
