# gpu_package_rules.py
GPU_REPLACEMENTS = {
    "torch": "torch-cpu",
    "tensorflow": "tensorflow-cpu",
    "pytorch": "pytorch-cpu",
    "cudatoolkit": None,
    "pytorch-cuda": None,
}

GPU_SUFFIXES = ["+cu117", "+cu118", "+cu121", "+cu122", "+cu123"]
