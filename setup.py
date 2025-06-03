from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()
    
setup(
    name="vox_profile",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=required,
    author="Tiantian Feng",
    author_email="tiantiaf@usc.edu",
    description="This repo includes Vox-Profile, one of the first benchmarking efforts that systematically evaluate rich multi-dimensional speaker and speech traits from English-speaking voices.",
    url='https://github.com/underdogliu/vox-profile-release',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
