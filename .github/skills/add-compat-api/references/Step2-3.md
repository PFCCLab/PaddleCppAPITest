# Step 2 参考：测试文件命名与 CMakeLists.txt 注册规范

本文用于补充 Step 2 的第 3 步：在 `$PADDLE_ROOT/test/cpp/compat` 中新增 compat 单测时的文件名与 CMake 注册规范。

## 测试文件名规范

新增测试文件应放在 `$PADDLE_ROOT/test/cpp/compat/` 目录下，命名统一为以下三种形式之一：

- `ATen_<OpName>_test.cc` —— ATen 算子接口测试（例：`ATen_chunk_test.cc`）
- `c10_<Feature>_test.cc` —— c10 基础设施测试（例：`c10_Stream_test.cc`）
- `torch_<Feature>_test.cc` —— torch 库接口测试（例：`torch_library_test.cc`）

## CMakeLists.txt 注册规则

每新增一个测试文件，必须同步在 `$PADDLE_ROOT/test/cpp/compat/CMakeLists.txt` 中注册。根据测试内容选择注册方式：

### 1. 纯 CPU 测试

文件内不含任何 CUDA 相关代码，可直接注册：

```cmake
cc_test(ATen_chunk_test SRCS ATen_chunk_test.cc)
```

### 2. 纯 CUDA 测试

文件内**仅含** CUDA 相关代码（调用 CUDA kernel、使用 CUDA runtime API 等），必须用 `nv_test` 注册，并包裹在 `if(WITH_GPU)` 条件下，确保无 CUDA 环境时不参与编译：

```cmake
if(WITH_GPU)
  nv_test(ATen_cuda_test SRCS ATen_cuda_test.cc)
endif()
```

### 3. 混合测试（同时含 CPU 与 CUDA 代码）

文件内既有 CPU 通用代码，又有 CUDA 专属代码路径，使用 `cc_test` 注册，但 CUDA 相关代码段必须用宏包裹，确保在无 CUDA 环境下仍能编译通过：

**CMakeLists.txt 注册：**

```cmake
cc_test(ATen_CUDAContext_test SRCS ATen_CUDAContext_test.cc)
```

**C++ 代码中 CUDA 段包裹：**

```cpp
// CPU 通用代码（无需宏）
TEST(ATen_CUDAContext, Basic) {
  // ...
}

// CUDA 专属代码段
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(ATen_CUDAContext, CUDAStream) {
  // 调用 CUDA runtime API ...
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
```

> 若测试同时需要适配 ROCm（HIP），宏条件应写为 `#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)`；若仅适配 CUDA，可简化为 `#if defined(PADDLE_WITH_CUDA)`。

## 快速检查清单

- [ ] 测试文件名符合 `ATen_<Op>_test.cc` / `c10_<Feature>_test.cc` / `torch_<Feature>_test.cc` 规范
- [ ] 已在 `$PADDLE_ROOT/test/cpp/compat/CMakeLists.txt` 中注册
- [ ] 纯 CUDA 测试使用了 `nv_test` + `if(WITH_GPU)`
- [ ] 混合测试中 CUDA 代码段已用 `#ifdef PADDLE_WITH_CUDA`（或含 `PADDLE_WITH_HIP`）宏包裹
- [ ] XPU / 纯 CPU 环境下 `ninja` 编译不会因此测试而失败
