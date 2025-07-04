"""Compatibility wrapper so tests can import the pipeline module."""

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys

src_path = Path(__file__).resolve().parents[1] / "src" / "pipeline.py"
spec = spec_from_file_location("_pipeline", src_path)
module = module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)  # type: ignore

globals().update(module.__dict__)
