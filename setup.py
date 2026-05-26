from setuptools import find_packages, setup


def _read_requirements(path):
    requirements = []
    with open(path) as f:
        for line in f:
            req = line.strip()
            if not req or req.startswith('#') or req.startswith('git+'):
                continue
            requirements.append(req)
    return requirements


inference_requirements = _read_requirements("requirements.inference.txt")
legacy_requirements = _read_requirements("requirements.training-legacy.txt")

setup(
    name="spurs",  
    version="1.0.0",
    description="",
    author="Ziang Li",
    author_email="zaing at gatech.edu",
    install_requires=inference_requirements,
    extras_require={
        "training-legacy": legacy_requirements,
    },
    packages=find_packages(),  
    package_dir={"": "."},  
)
