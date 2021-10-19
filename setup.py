import setuptools


setuptools.setup(
    name='punctuation_corrector',
    version='0.0.1',
    description='Simple punctuation correction tool',
    packages=[
        'punctuation_corrector',
        'punctuation_corrector.common',
        'punctuation_corrector.data_sources',
        'punctuation_corrector.inference',
        'punctuation_corrector.training'
    ],
    license='MIT',
    install_requires=[
        'numpy>=1.19.5', 'transformers>=4.11.2', 'onnx>=1.10.1', 'onnxruntime>=1.9.0'],
    extras_require={
        'dev': [
            'scikit-learn>=0.24.2', 'torch>=1.9.1', 'pytorch-lightning>=1.4.2', 'pandas>=1.1.5',
            'lxml>=4.6.3'
        ]
    }
)
