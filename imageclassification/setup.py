from setuptools import setup


def _read(f) -> bytes:
    """
    Reads in the content of the file.
    :param f: the file to read
    :type f: str
    :return: the content
    :rtype: str
    """
    return open(f, 'rb').read()


setup(
    name="wai-keras-imaging",
    description="Python library for performing image classification with Keras.",
    long_description=(
        _read('DESCRIPTION.rst') + b'\n' +
        _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/waikato-datamining/keras-imaging",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Internet :: WWW/HTTP',
        'Programming Language :: Python :: 3',
    ],
    license='MIT',
    package_dir={
        '': 'src'
    },
    packages=[
        'predict',
        'train',
    ],
    version="0.0.1",
    author='Peter Reutemann',
    author_email='fracpete@waikato.ac.nz',
    install_requires=[
        "tensorflow",
        "keras",
        "pandas",
        "pillow",
        "matplotlib",
        "wai.annotations.voc",
        "wai.annotations.coco"
    ],
    include_package_data=True
)
