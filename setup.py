from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["tqdm>=4.32.1"]

setup(
    name="notebookc",
    python_requires='3.6.0',
    version="0.0.1",
    author="Aurélien COLIN",
    author_email="acolin@groupcls.com",
    description="Work on mosaic with overlapping tiles",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Rignak/MightyMosaic/homepage/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
