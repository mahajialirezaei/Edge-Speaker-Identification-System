try:
    import sherpa_onnx
    import numpy as np
    print("🚀 environment setup successfully!")
    print(f"Sherpa-ONNX Version: {sherpa_onnx.__version__}")
except ImportError as e:
    print(f"❌ Error: {e}")