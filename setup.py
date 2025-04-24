from setuptools import find_packages, setup

# Read requirements.txt and filter out git URLs
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if not line.strip().startswith('git+')]

setup(
    name="spurs",  
    version="1.0.0",
    description="",
    author="Ziang Li",
    author_email="zaing at gatech.edu",
    install_requires=requirements,
    packages=find_packages(),  
    package_dir={"": "."},  
)
