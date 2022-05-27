import itertools
import os

from setuptools import find_packages, setup

package_name = "srl"

# read version
version_path = os.path.join(os.path.dirname(__file__), package_name, "version.py")
exec(open(version_path).read())

extras = {
    "gym": ["gym", "pygame"],
    "image": ["matplotlib", "opencv-python", "pillow", "pandas"],
    "notebook": ["matplotlib", "opencv-python", "pillow", "pandas", "ipython"],
    "kaggle": ["kaggle_environments"],
    "spec": [
        "psutil",  # CPU info
        "pynvml",  # GPU info
    ],
}
extras["all"] = list(set(itertools.chain.from_iterable([arr for arr in extras.values()])))

setup(
    name=package_name,
    packages=[package for package in find_packages() if package.startswith(package_name)],
    version=VERSION,
    license="MIT",
    author="poco",
    author_email="pocopococpp198@gmail.com",
    url="https://github.com/pocokhc/simple_distributed_rl",
    description="simple distributed reinforcement learning framework",
    long_description="simple distributed reinforcement learning framework",
    install_requires=[
        "numpy",
        "tensorflow",
        "tensorflow-addons",
    ],
    extras_require=extras,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
