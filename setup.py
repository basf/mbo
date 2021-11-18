import os.path

from setuptools import find_packages, setup


def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    fp = os.path.join(here, "mbo/__init__.py")
    for line in open(fp).readlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    return ""


root_dir = os.path.dirname(__file__)
with open(os.path.join(root_dir, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="mbo",
    description="Multiobjective Bayesian Optimization",
    author="BASF SE",
    license="BSD-3",
    url="https://github.com/basf/mbo",
    keywords=[
        "Bayesian optimization",
        "Multi-objective optimization",
        "Experimental design",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=get_version(),
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["mopti>=0.1.0", "numpy", "pandas"],
    extras_require={"testing": ["pytest"]},
)
