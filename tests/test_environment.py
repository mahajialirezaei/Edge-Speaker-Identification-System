import sherpa_onnx
import numpy as np

def test_sherpa_onnx_version():
    assert hasattr(sherpa_onnx, '__version__'), "sherpa_onnx version attribute is missing"

def test_numpy_import():
    assert np.__name__ == 'numpy'