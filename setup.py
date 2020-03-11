from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["tqdm>=4.32.1"]

setup(
    name="MightyMosaic",
    python_requires='>=3.6',
    version="1.2.2",
    author="Aurélien COLIN",
    author_email="aureliencolin@hotmail.com",
    description="Create mosaics with overlapping tiles",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Rignak/MightyMosaic",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
