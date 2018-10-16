import os.path as op

import setuptools

readme = op.join(op.abspath(op.dirname(__file__)), "README.rst")
with open(readme, "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hybridfactory",
    version="0.1.0-beta1",
    packages=setuptools.find_packages(),
    url="https://gitlab.com/vidriotech/spiegel/hybridfactory",
    license="MIT",
    author="Alan Liddell",
    author_email="alan@vidriotech.com",
    description="A utility for generating hybrid ephys data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "hybridfactory = hybridfactory.utils.cli:main"
        ]
    }
)
