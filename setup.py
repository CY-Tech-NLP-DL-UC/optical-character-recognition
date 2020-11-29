from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ocrapi",
    version="0.0",
    description="OCR API for fun",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "flask",
        "python-dotenv",
        "torch",
        "torchvision",
        "Pillow",
        "opencv-python",
        "tensorflow",
        "Keras",
    ],
    packages=find_packages(),
)
