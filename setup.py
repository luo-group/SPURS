from setuptools import find_packages, setup

setup(
    name="spurs",  
    version="1.0.0",
    description="",
    author="Ziang Li",
    author_email="zaing at gatech.edu",
    install_requires=open("requirements.txt").readlines(),
    packages=find_packages(),  
    package_dir={"": "."},  
)
