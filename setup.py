from setuptools import setup
import codecs
from codecs import open
from os import path

package_name = "cannai"
PACKAGE      = package_name

PROJECT = path.abspath(path.dirname(__file__))

root_dir = path.abspath(path.dirname(__file__))

def _requirements():
    return [name.rstrip() for name in open(path.join(root_dir, 'requirements.txt')).readlines()]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

VERSION_PATH = path.join(PACKAGE, "version.py")

def read(*parts):
    """
    Assume UTF-8 encoding and return the contents of the file located at the
    absolute path from the REPOSITORY joined with *parts.
    """
    with codecs.open(path.join(PROJECT, *parts), "rb", "utf-8") as f:
        return f.read()

def get_version(path=VERSION_PATH):
    """
    Reads the python file defined in the VERSION_PATH to find the get_version
    function, and executes it to ensure that it is loaded correctly. Separating
    the version in this way ensures no additional code is executed.
    """
    namespace = {}
    exec(read(path), namespace)
    return namespace["get_version"](short=True)

REQUIRE_PATH = "requirements.txt"

def get_requires(path=REQUIRE_PATH):
    """
    Yields a generator of requirements as defined by the REQUIRE_PATH which
    should point to a requirements.txt output by `pip freeze`.
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line

setup(
    name=package_name,
    version=get_version(),
    description='data visualization for machine learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    #url='Github等のurl',
    author='Ryo Kinoshita',
    author_email='Rkinoshi55@gmail.com',
    license='MIT',
    keywords='machinelearning visualization graph',
    packages=[package_name,package_name + ".model_compare"],
    install_requires=list(get_requires()),
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Framework :: Matplotlib',
    ],
)