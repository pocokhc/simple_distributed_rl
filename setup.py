import os

from setuptools import find_packages, setup

package_name = "srl"
here = os.path.abspath(os.path.dirname(__file__))

# read version
version_path = os.path.join(here, package_name, "version.py")
exec(open(version_path).read())

setup(
    name=package_name,
    packages=[package for package in find_packages() if package.startswith(package_name)],
    package_data={
        "srl": [
            "font/*.ttf",
            "font/README.md",
            "envs/img/*.png",
            "envs/img/README.md",
            "runner/img/*.svg",
            "runner/img/README.md",
        ],
    },
    version=VERSION,  # type: ignore  # noqa
    license="MIT",
    author="poco",
    author_email="pocopococpp198@gmail.com",
    url="https://github.com/pocokhc/simple_distributed_rl",
    description="simple distributed reinforcement learning framework",
    long_description=open(os.path.join(here, "README.md"), encoding="utf-8").read().replace("\r", ""),
    long_description_content_type="text/markdown",
    install_requires=["numpy"],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
