import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hybridfactory",
    version="0.1.2",
    packages=setuptools.find_packages(),
    url="https://gitlab.com/vidriotech/spiegel/hybridfactory",
    license="",
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
            "generate = hybridfactory.utils.cli:main"
        ]
    }
)
