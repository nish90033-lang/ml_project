from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path):
    with open(file_path) as file:
        requirements = file.read().splitlines()
    return [req for req in requirements if req != HYPHEN_E_DOT]
setup(
    name='mlproject',
    version='0.0.1',
    author='Nishant',
    author_email='nish90033@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)