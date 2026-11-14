from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PascalCompatibilityProfileTests(unittest.TestCase):
    def test_setup_selects_python_and_tensorflow_compatible_with_pascal(self):
        setup = (ROOT / "cli" / "setup_python_venv").read_text(encoding="utf-8")
        self.assertIn("6.1) IS_PASCAL=true", setup)
        self.assertIn("PYTHONS=( python3.10 )", setup)
        self.assertIn('DEFAULT_TF_SPEC="tensorflow==2.10.1"', setup)
        self.assertIn('pip_install "numpy==1.23.5"', setup)
        self.assertIn('pip_install "keras==2.10.0"', setup)

    def test_setup_selects_pascal_torch_and_avoids_cuda_onnxruntime(self):
        setup = (ROOT / "cli" / "setup_python_venv").read_text(encoding="utf-8")
        self.assertIn('MWW_TORCH_CUDA:-cu124', setup)
        self.assertIn('MWW_TORCH_VERSION:-2.5.0', setup)
        self.assertIn('if ${GPU} && ! ${IS_PASCAL} ; then', setup)

    def test_pascal_image_and_external_trainers_use_compatible_frameworks(self):
        dockerfile = (ROOT / "dockerfile.pascal").read_text(encoding="utf-8")
        self.assertIn("11.2.2-cudnn8-runtime-ubuntu20.04", dockerfile)
        self.assertIn("MWW_PASCAL_IMAGE=1", dockerfile)
        self.assertIn("PYTHON=python3.10", dockerfile)
        self.assertIn("NWW_TORCH_VERSION=2.5.0", dockerfile)
        self.assertIn("OWW_TORCH_VERSION=2.5.0", dockerfile)


if __name__ == "__main__":
    unittest.main()
