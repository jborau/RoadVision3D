from setuptools import setup, find_packages

setup(
    name="roadvision3d",
    version="0.1.0",
    packages=find_packages(),  # Make sure this correctly finds all sub-packages
    install_requires=[
        "numpy",
        "torch",  # Assuming you're using PyTorch
        "opencv-python",
        "PyYAML"
    ],
    entry_points={
        'console_scripts': [
            'roadvision3d_train=roadvision3d.src.tools.train_val:main',
        ],
    },
)
