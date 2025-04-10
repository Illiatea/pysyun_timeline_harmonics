from setuptools import setup

setup(
    name='pysyun_timeline_harmonics',
    version='1.1.3',
    author='Illiatea',
    author_email='illiatea2@gmail.com',
    py_modules=['fourier_intensity_processor', 'circade', 'renders'],
    install_requires=['scipy', 'numpy', 'matplotlib']
)
