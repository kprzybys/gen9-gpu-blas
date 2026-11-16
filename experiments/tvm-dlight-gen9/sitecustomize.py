# Put this directory on PYTHONPATH while running `mlc_llm compile` for Intel OpenCL targets:
# registers the __local hoisting post-processor (opencl_local_hoist.py) before TVM generates code.
try:
    import opencl_local_hoist  # noqa: F401
except ImportError:  # tvm_ffi not available in this interpreter
    pass
