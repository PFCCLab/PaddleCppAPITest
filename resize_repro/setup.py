import paddle

paddle.enable_compat()

import torch  # noqa: F401
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)

setup(
    name="resize_repro",
    ext_modules=[
        CUDAExtension(
            name="resize_repro",
            sources=["resize_repro.cpp"],
            extra_compile_args={
                "cxx": ["-DPADDLE_WITH_CUDA"],
                "nvcc": [],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
