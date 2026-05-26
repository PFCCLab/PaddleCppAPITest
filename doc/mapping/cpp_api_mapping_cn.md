# PyTorch C++ API (libtorch) 与 Paddle C++ API 映射表

本文梳理了 PyTorch C++ API (libtorch) 与 PaddlePaddle C++ API 的对应关系与差异分析，
帮助开发者快速迁移 PyTorch C++ 使用经验。

> **Note**: 本映射表基于以下路径**自动解析 C++ 函数签名**生成：
> - PyTorch C++ API 头文件: `D:/Lenovo/libtorch/include/ATen/ops`
> - Paddle compat 层头文件: `D:/Lenovo/Paddle/paddle/phi/api/include/compat/ATen/ops`
> - Paddle `api.h` 头文件: `D:/Lenovo/Paddle/paddle/phi/api/include/api.h`

> **说明**: 对于 compat 层未封装的函数，脚本直接对比 libtorch 头文件与 `paddle::experimental` 命名空间中同名函数的**返回类型、参数类型、参数名、参数默认值、参数数量**，按优先级自动归入差异分类。

## API 映射分类

| 序号 | 类别 | 简介 |
| ---- | ---- | ---- |
| 1 | API 完全一致 | compat 层已实现与 PyTorch C++ API 完全一致的接口，可直接替换命名空间使用 |
| 2 | 仅 API 调用方式不一致 | Paddle 有同名实现，但调用方式与 PyTorch 不一致（签名解析后兜底分类） |
| 3 | 仅参数名不一致 | 功能相同，但部分参数名称不同 |
| 4 | paddle 参数更多 | Paddle 中提供了更多可选参数 |
| 5 | 参数默认值不一致 | 功能相同，但某些参数的默认值不同 |
| 6 | torch 参数更多 | PyTorch 中提供了更多参数 |
| 7 | 输入参数用法不一致 | 对输入参数的处理方式不同 |
| 8 | 输入参数类型不一致 | 要求的输入数据类型不同 |
| 9 | 返回参数类型不一致 | 返回值的类型或结构不同 |
| 10 | 组合替代实现 | 在 Paddle 中没有直接对应的单一 API，需要多个 API 组合实现 |
| 11 | API 别名 | PyTorch 与 Paddle 功能一致，但 API 名称不同 |
| 12 | 功能缺失 | PyTorch C++ API 的功能在 Paddle 中暂时没有等效实现 |

### 1. API 完全一致

**简介：** compat 层已实现与 PyTorch C++ API 完全一致的接口，只需将代码中的命名空间或调用方式按 compat 层声明使用即可。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::_local_scalar_dense` | `at::_local_scalar_dense` (compat层) | API 完全一致 | - |
| 2 | `at::_nnz` | `at::_nnz` (compat层) | API 完全一致 | - |
| 3 | `at::_values` | `at::_values` (compat层) | API 完全一致 | - |
| 4 | `at::abs` | `at::abs` (compat层) | API 完全一致 | - |
| 5 | `at::all` | `at::all` (compat层) | API 完全一致 | - |
| 6 | `at::allclose` | `at::allclose` (compat层) | API 完全一致 | - |
| 7 | `at::any` | `at::any` (compat层) | API 完全一致 | - |
| 8 | `at::arange` | `at::arange` (compat层) | API 完全一致 | - |
| 9 | `at::as_strided` | `at::as_strided` (compat层) | API 完全一致 | - |
| 10 | `at::cat` | `at::cat` (compat层) | API 完全一致 | - |
| 11 | `at::chunk` | `at::chunk` (compat层) | API 完全一致 | - |
| 12 | `at::clamp` | `at::clamp` (compat层) | API 完全一致 | - |
| 13 | `at::coalesce` | `at::coalesce` (compat层) | API 完全一致 | - |
| 14 | `at::detach` | `at::detach` (compat层) | API 完全一致 | - |
| 15 | `at::dsplit` | `at::dsplit` (compat层) | API 完全一致 | - |
| 16 | `at::empty` | `at::empty` (compat层) | API 完全一致 | - |
| 17 | `at::empty_like` | `at::empty_like` (compat层) | API 完全一致 | - |
| 18 | `at::empty_strided` | `at::empty_strided` (compat层) | API 完全一致 | - |
| 19 | `at::equal` | `at::equal` (compat层) | API 完全一致 | - |
| 20 | `at::expand` | `at::expand` (compat层) | API 完全一致 | - |
| 21 | `at::eye` | `at::eye` (compat层) | API 完全一致 | - |
| 22 | `at::flatten` | `at::flatten` (compat层) | API 完全一致 | - |
| 23 | `at::from_blob` | `at::from_blob` (compat层) | API 完全一致 | - |
| 24 | `at::full` | `at::full` (compat层) | API 完全一致 | - |
| 25 | `at::hsplit` | `at::hsplit` (compat层) | API 完全一致 | - |
| 26 | `at::index` | `at::index` (compat层) | API 完全一致 | - |
| 27 | `at::index_put` | `at::index_put` (compat层) | API 完全一致 | - |
| 28 | `at::is_coalesced` | `at::is_coalesced` (compat层) | API 完全一致 | - |
| 29 | `at::item` | `at::item` (compat层) | API 完全一致 | - |
| 30 | `at::masked_select` | `at::masked_select` (compat层) | API 完全一致 | - |
| 31 | `at::narrow` | `at::narrow` (compat层) | API 完全一致 | - |
| 32 | `at::narrow_copy` | `at::narrow_copy` (compat层) | API 完全一致 | - |
| 33 | `at::new_empty` | `at::new_empty` (compat层) | API 完全一致 | - |
| 34 | `at::new_full` | `at::new_full` (compat层) | API 完全一致 | - |
| 35 | `at::new_ones` | `at::new_ones` (compat层) | API 完全一致 | - |
| 36 | `at::new_zeros` | `at::new_zeros` (compat层) | API 完全一致 | - |
| 37 | `at::ones` | `at::ones` (compat层) | API 完全一致 | - |
| 38 | `at::permute` | `at::permute` (compat层) | API 完全一致 | - |
| 39 | `at::reciprocal` | `at::reciprocal` (compat层) | API 完全一致 | - |
| 40 | `at::record_stream` | `at::record_stream` (compat层) | API 完全一致 | - |
| 41 | `at::rename` | `at::rename` (compat层) | API 完全一致 | - |
| 42 | `at::reshape` | `at::reshape` (compat层) | API 完全一致 | - |
| 43 | `at::resize` | `at::resize` (compat层) | API 完全一致 | - |
| 44 | `at::select` | `at::select` (compat层) | API 完全一致 | - |
| 45 | `at::slice` | `at::slice` (compat层) | API 完全一致 | - |
| 46 | `at::sparse_coo_tensor` | `at::sparse_coo_tensor` (compat层) | API 完全一致 | - |
| 47 | `at::sparse_csr_tensor` | `at::sparse_csr_tensor` (compat层) | API 完全一致 | - |
| 48 | `at::split` | `at::split` (compat层) | API 完全一致 | - |
| 49 | `at::split_with_sizes` | `at::split_with_sizes` (compat层) | API 完全一致 | - |
| 50 | `at::squeeze` | `at::squeeze` (compat层) | API 完全一致 | - |
| 51 | `at::std` | `at::std` (compat层) | API 完全一致 | - |
| 52 | `at::sum` | `at::sum` (compat层) | API 完全一致 | - |
| 53 | `at::t` | `at::t` (compat层) | API 完全一致 | - |
| 54 | `at::tensor` | `at::tensor` (compat层) | API 完全一致 | - |
| 55 | `at::tensor_split` | `at::tensor_split` (compat层) | API 完全一致 | - |
| 56 | `at::to` | `at::to` (compat层) | API 完全一致 | - |
| 57 | `at::transpose` | `at::transpose` (compat层) | API 完全一致 | - |
| 58 | `at::unflatten` | `at::unflatten` (compat层) | API 完全一致 | - |
| 59 | `at::unsafe_split` | `at::unsafe_split` (compat层) | API 完全一致 | - |
| 60 | `at::unsafe_split_with_sizes` | `at::unsafe_split_with_sizes` (compat层) | API 完全一致 | - |
| 61 | `at::unsqueeze` | `at::unsqueeze` (compat层) | API 完全一致 | - |
| 62 | `at::view` | `at::view` (compat层) | API 完全一致 | - |
| 63 | `at::view_as` | `at::view_as` (compat层) | API 完全一致 | - |
| 64 | `at::vsplit` | `at::vsplit` (compat层) | API 完全一致 | - |
| 65 | `at::zeros` | `at::zeros` (compat层) | API 完全一致 | - |
| 66 | `at::zeros_like` | `at::zeros_like` (compat层) | API 完全一致 | - |

### 2. 仅 API 调用方式不一致

**简介：** Paddle `paddle::experimental` 命名空间中有同名实现，但 compat 层尚未提供完全一致的封装。以下函数经签名对比后归入此类，多为调用语义或底层实现差异。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::broadcast_tensors` | `paddle::experimental::broadcast_tensors` | 仅 API 调用方式不一致 | 签名高度相似，调用方式或语义有细微差异 |
| 2 | `at::complex` | `paddle::experimental::complex` | 仅 API 调用方式不一致 | 签名高度相似，调用方式或语义有细微差异 |
| 3 | `at::meshgrid` | `paddle::experimental::meshgrid` | 仅 API 调用方式不一致 | 签名高度相似，调用方式或语义有细微差异 |
| 4 | `at::nonzero` | `paddle::experimental::nonzero` | 仅 API 调用方式不一致 | 签名高度相似，调用方式或语义有细微差异 |

### 3. 仅参数名不一致

**简介：** 此类 API 功能相同，但部分参数名称不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::acos` | `paddle::experimental::acos` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.acos.md) |
| 2 | `at::acosh` | `paddle::experimental::acosh` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.acosh.md) |
| 3 | `at::amax` | `paddle::experimental::amax` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.amax.md) |
| 4 | `at::amin` | `paddle::experimental::amin` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.amin.md) |
| 5 | `at::angle` | `paddle::experimental::angle` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.angle.md) |
| 6 | `at::asin` | `paddle::experimental::asin` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.asin.md) |
| 7 | `at::asinh` | `paddle::experimental::asinh` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.asinh.md) |
| 8 | `at::atan` | `paddle::experimental::atan` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.atan.md) |
| 9 | `at::atan2` | `paddle::experimental::atan2` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.atan2.md) |
| 10 | `at::atanh` | `paddle::experimental::atanh` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.atanh.md) |
| 11 | `at::bitwise_not` | `paddle::experimental::bitwise_not` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.bitwise_not.md) |
| 12 | `at::bmm` | `paddle::experimental::bmm` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.bmm.md) |
| 13 | `at::ceil` | `paddle::experimental::ceil` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.ceil.md) |
| 14 | `at::cholesky` | `paddle::experimental::cholesky` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.cholesky.md) |
| 15 | `at::cholesky_solve` | `paddle::experimental::cholesky_solve` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.cholesky_solve.md) |
| 16 | `at::conj` | `paddle::experimental::conj` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.conj.md) |
| 17 | `at::copysign` | `paddle::experimental::copysign` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.copysign.md) |
| 18 | `at::cos` | `paddle::experimental::cos` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.cos.md) |
| 19 | `at::cosh` | `paddle::experimental::cosh` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.cosh.md) |
| 20 | `at::det` | `paddle::experimental::det` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.det.md) |
| 21 | `at::digamma` | `paddle::experimental::digamma` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.digamma.md) |
| 22 | `at::divide` | `paddle::experimental::divide` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.divide.md) |
| 23 | `at::dot` | `paddle::experimental::dot` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.dot.md) |
| 24 | `at::erf` | `paddle::experimental::erf` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.erf.md) |
| 25 | `at::erfinv` | `paddle::experimental::erfinv` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.erfinv.md) |
| 26 | `at::exp` | `paddle::experimental::exp` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.exp.md) |
| 27 | `at::expm1` | `paddle::experimental::expm1` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.expm1.md) |
| 28 | `at::floor` | `paddle::experimental::floor` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.floor.md) |
| 29 | `at::floor_divide` | `paddle::experimental::floor_divide` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.floor_divide.md) |
| 30 | `at::fmax` | `paddle::experimental::fmax` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.fmax.md) |
| 31 | `at::fmin` | `paddle::experimental::fmin` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.fmin.md) |
| 32 | `at::full_like` | `paddle::experimental::full_like` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.full_like.md) |
| 33 | `at::hardshrink` | `paddle::experimental::hardshrink` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.hardshrink.md) |
| 34 | `at::hardswish` | `paddle::experimental::hardswish` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.hardswish.md) |
| 35 | `at::hardtanh` | `paddle::experimental::hardtanh` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.hardtanh.md) |
| 36 | `at::heaviside` | `paddle::experimental::heaviside` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.heaviside.md) |
| 37 | `at::i0` | `paddle::experimental::i0` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.i0.md) |
| 38 | `at::imag` | `paddle::experimental::imag` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.imag.md) |
| 39 | `at::inverse` | `paddle::experimental::inverse` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.inverse.md) |
| 40 | `at::isfinite` | `paddle::experimental::isfinite` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.isfinite.md) |
| 41 | `at::isinf` | `paddle::experimental::isinf` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.isinf.md) |
| 42 | `at::isnan` | `paddle::experimental::isnan` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.isnan.md) |
| 43 | `at::kron` | `paddle::experimental::kron` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.kron.md) |
| 44 | `at::lgamma` | `paddle::experimental::lgamma` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.lgamma.md) |
| 45 | `at::log` | `paddle::experimental::log` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.log.md) |
| 46 | `at::log10` | `paddle::experimental::log10` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.log10.md) |
| 47 | `at::log1p` | `paddle::experimental::log1p` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.log1p.md) |
| 48 | `at::log2` | `paddle::experimental::log2` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.log2.md) |
| 49 | `at::logical_and` | `paddle::experimental::logical_and` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.logical_and.md) |
| 50 | `at::logical_not` | `paddle::experimental::logical_not` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.logical_not.md) |
| 51 | `at::logical_or` | `paddle::experimental::logical_or` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.logical_or.md) |
| 52 | `at::logical_xor` | `paddle::experimental::logical_xor` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.logical_xor.md) |
| 53 | `at::masked_scatter` | `paddle::experimental::masked_scatter` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.masked_scatter.md) |
| 54 | `at::maximum` | `paddle::experimental::maximum` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.maximum.md) |
| 55 | `at::minimum` | `paddle::experimental::minimum` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.minimum.md) |
| 56 | `at::multiply` | `paddle::experimental::multiply` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.multiply.md) |
| 57 | `at::mv` | `paddle::experimental::mv` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.mv.md) |
| 58 | `at::nextafter` | `paddle::experimental::nextafter` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.nextafter.md) |
| 59 | `at::ones_like` | `paddle::experimental::ones_like` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.ones_like.md) |
| 60 | `at::pow` | `paddle::experimental::pow` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.pow.md) |
| 61 | `at::real` | `paddle::experimental::real` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.real.md) |
| 62 | `at::relu` | `paddle::experimental::relu` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.relu.md) |
| 63 | `at::relu6` | `paddle::experimental::relu6` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.relu6.md) |
| 64 | `at::rsqrt` | `paddle::experimental::rsqrt` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.rsqrt.md) |
| 65 | `at::scatter` | `paddle::experimental::scatter` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.scatter.md) |
| 66 | `at::sigmoid` | `paddle::experimental::sigmoid` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.sigmoid.md) |
| 67 | `at::sign` | `paddle::experimental::sign` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.sign.md) |
| 68 | `at::silu` | `paddle::experimental::silu` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.silu.md) |
| 69 | `at::sin` | `paddle::experimental::sin` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.sin.md) |
| 70 | `at::sinh` | `paddle::experimental::sinh` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.sinh.md) |
| 71 | `at::softshrink` | `paddle::experimental::softshrink` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.softshrink.md) |
| 72 | `at::sqrt` | `paddle::experimental::sqrt` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.sqrt.md) |
| 73 | `at::square` | `paddle::experimental::square` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.square.md) |
| 74 | `at::tan` | `paddle::experimental::tan` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.tan.md) |
| 75 | `at::tanh` | `paddle::experimental::tanh` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.tanh.md) |
| 76 | `at::tile` | `paddle::experimental::tile` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.tile.md) |
| 77 | `at::trunc` | `paddle::experimental::trunc` | 仅参数名不一致 | [差异对比](cpp_args_name_diff/at.trunc.md) |

### 4. paddle 参数更多

**简介：** 此类 API 在 Paddle 中提供了更多可选参数。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::argmax` | `paddle::experimental::argmax` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.argmax.md) |
| 2 | `at::argmin` | `paddle::experimental::argmin` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.argmin.md) |
| 3 | `at::baddbmm` | `paddle::experimental::baddbmm` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.baddbmm.md) |
| 4 | `at::bitwise_left_shift` | `paddle::experimental::bitwise_left_shift` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.bitwise_left_shift.md) |
| 5 | `at::bitwise_right_shift` | `paddle::experimental::bitwise_right_shift` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.bitwise_right_shift.md) |
| 6 | `at::channel_shuffle` | `paddle::experimental::channel_shuffle` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.channel_shuffle.md) |
| 7 | `at::conv2d` | `paddle::experimental::conv2d` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.conv2d.md) |
| 8 | `at::conv3d` | `paddle::experimental::conv3d` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.conv3d.md) |
| 9 | `at::cumprod` | `paddle::experimental::cumprod` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.cumprod.md) |
| 10 | `at::cumsum` | `paddle::experimental::cumsum` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.cumsum.md) |
| 11 | `at::diag` | `paddle::experimental::diag` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.diag.md) |
| 12 | `at::dropout` | `paddle::experimental::dropout` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.dropout.md) |
| 13 | `at::frobenius_norm` | `paddle::experimental::frobenius_norm` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.frobenius_norm.md) |
| 14 | `at::hardsigmoid` | `paddle::experimental::hardsigmoid` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.hardsigmoid.md) |
| 15 | `at::linspace` | `paddle::experimental::linspace` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.linspace.md) |
| 16 | `at::logcumsumexp` | `paddle::experimental::logcumsumexp` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.logcumsumexp.md) |
| 17 | `at::logspace` | `paddle::experimental::logspace` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.logspace.md) |
| 18 | `at::logsumexp` | `paddle::experimental::logsumexp` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.logsumexp.md) |
| 19 | `at::lu_solve` | `paddle::experimental::lu_solve` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.lu_solve.md) |
| 20 | `at::matmul` | `paddle::experimental::matmul` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.matmul.md) |
| 21 | `at::max` | `paddle::experimental::max` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.max.md) |
| 22 | `at::mean` | `paddle::experimental::mean` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.mean.md) |
| 23 | `at::min` | `paddle::experimental::min` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.min.md) |
| 24 | `at::mish` | `paddle::experimental::mish` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.mish.md) |
| 25 | `at::pixel_shuffle` | `paddle::experimental::pixel_shuffle` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.pixel_shuffle.md) |
| 26 | `at::pixel_unshuffle` | `paddle::experimental::pixel_unshuffle` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.pixel_unshuffle.md) |
| 27 | `at::prelu` | `paddle::experimental::prelu` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.prelu.md) |
| 28 | `at::prod` | `paddle::experimental::prod` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.prod.md) |
| 29 | `at::randint` | `paddle::experimental::randint` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.randint.md) |
| 30 | `at::random` | `paddle::experimental::random` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.random.md) |
| 31 | `at::randperm` | `paddle::experimental::randperm` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.randperm.md) |
| 32 | `at::repeat_interleave` | `paddle::experimental::repeat_interleave` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.repeat_interleave.md) |
| 33 | `at::round` | `paddle::experimental::round` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.round.md) |
| 34 | `at::selu` | `paddle::experimental::selu` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.selu.md) |
| 35 | `at::set` | `paddle::experimental::set` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.set.md) |
| 36 | `at::trace` | `paddle::experimental::trace` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.trace.md) |
| 37 | `at::tril_indices` | `paddle::experimental::tril_indices` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.tril_indices.md) |
| 38 | `at::triu_indices` | `paddle::experimental::triu_indices` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.triu_indices.md) |
| 39 | `at::var` | `paddle::experimental::var` | paddle 参数更多 | [差异对比](cpp_paddle_more_args/at.var.md) |

### 5. 参数默认值不一致

**简介：** 此类 API 功能相同，但某些参数的默认值不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::fill` | `paddle::experimental::fill` | 参数默认值不一致 | [差异对比](cpp_args_default_value_diff/at.fill.md) |
| 2 | `at::roll` | `paddle::experimental::roll` | 参数默认值不一致 | [差异对比](cpp_args_default_value_diff/at.roll.md) |

### 6. torch 参数更多

**简介：** 此类 API 在 PyTorch 中提供了更多参数。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::add` | `paddle::experimental::add` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.add.md) |
| 2 | `at::bernoulli` | `paddle::experimental::bernoulli` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.bernoulli.md) |
| 3 | `at::binomial` | `paddle::experimental::binomial` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.binomial.md) |
| 4 | `at::elu` | `paddle::experimental::elu` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.elu.md) |
| 5 | `at::embedding` | `paddle::experimental::embedding` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.embedding.md) |
| 6 | `at::gather` | `paddle::experimental::gather` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.gather.md) |
| 7 | `at::huber_loss` | `paddle::experimental::huber_loss` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.huber_loss.md) |
| 8 | `at::index_add` | `paddle::experimental::index_add` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.index_add.md) |
| 9 | `at::instance_norm` | `paddle::experimental::instance_norm` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.instance_norm.md) |
| 10 | `at::layer_norm` | `paddle::experimental::layer_norm` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.layer_norm.md) |
| 11 | `at::log_softmax` | `paddle::experimental::log_softmax` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.log_softmax.md) |
| 12 | `at::multinomial` | `paddle::experimental::multinomial` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.multinomial.md) |
| 13 | `at::pad` | `paddle::experimental::pad` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.pad.md) |
| 14 | `at::poisson` | `paddle::experimental::poisson` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.poisson.md) |
| 15 | `at::rrelu` | `paddle::experimental::rrelu` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.rrelu.md) |
| 16 | `at::searchsorted` | `paddle::experimental::searchsorted` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.searchsorted.md) |
| 17 | `at::softmax` | `paddle::experimental::softmax` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.softmax.md) |
| 18 | `at::stft` | `paddle::experimental::stft` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.stft.md) |
| 19 | `at::subtract` | `paddle::experimental::subtract` | torch 参数更多 | [差异对比](cpp_torch_more_args/at.subtract.md) |

### 7. 输入参数用法不一致

**简介：** 此类 API 对输入参数的处理方式不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| - | - | - | - | 暂无 |

### 8. 输入参数类型不一致

**简介：** 此类 API 要求的输入数据类型不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::addmm` | `paddle::experimental::addmm` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.addmm.md) |
| 2 | `at::bilinear` | `paddle::experimental::bilinear` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.bilinear.md) |
| 3 | `at::bincount` | `paddle::experimental::bincount` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.bincount.md) |
| 4 | `at::bitwise_and` | `paddle::experimental::bitwise_and` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.bitwise_and.md) |
| 5 | `at::bitwise_or` | `paddle::experimental::bitwise_or` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.bitwise_or.md) |
| 6 | `at::bitwise_xor` | `paddle::experimental::bitwise_xor` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.bitwise_xor.md) |
| 7 | `at::celu` | `paddle::experimental::celu` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.celu.md) |
| 8 | `at::clip` | `paddle::experimental::clip` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.clip.md) |
| 9 | `at::concat` | `paddle::experimental::concat` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.concat.md) |
| 10 | `at::cross` | `paddle::experimental::cross` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.cross.md) |
| 11 | `at::diag_embed` | `paddle::experimental::diag_embed` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.diag_embed.md) |
| 12 | `at::diagonal` | `paddle::experimental::diagonal` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.diagonal.md) |
| 13 | `at::dist` | `paddle::experimental::dist` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.dist.md) |
| 14 | `at::flip` | `paddle::experimental::flip` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.flip.md) |
| 15 | `at::gelu` | `paddle::experimental::gelu` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.gelu.md) |
| 16 | `at::greater_equal` | `paddle::experimental::greater_equal` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.greater_equal.md) |
| 17 | `at::group_norm` | `paddle::experimental::group_norm` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.group_norm.md) |
| 18 | `at::index_fill` | `paddle::experimental::index_fill` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.index_fill.md) |
| 19 | `at::index_select` | `paddle::experimental::index_select` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.index_select.md) |
| 20 | `at::isclose` | `paddle::experimental::isclose` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.isclose.md) |
| 21 | `at::leaky_relu` | `paddle::experimental::leaky_relu` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.leaky_relu.md) |
| 22 | `at::lerp` | `paddle::experimental::lerp` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.lerp.md) |
| 23 | `at::less_equal` | `paddle::experimental::less_equal` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.less_equal.md) |
| 24 | `at::logit` | `paddle::experimental::logit` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.logit.md) |
| 25 | `at::masked_fill` | `paddle::experimental::masked_fill` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.masked_fill.md) |
| 26 | `at::matrix_power` | `paddle::experimental::matrix_power` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.matrix_power.md) |
| 27 | `at::nansum` | `paddle::experimental::nansum` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.nansum.md) |
| 28 | `at::not_equal` | `paddle::experimental::not_equal` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.not_equal.md) |
| 29 | `at::one_hot` | `paddle::experimental::one_hot` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.one_hot.md) |
| 30 | `at::polygamma` | `paddle::experimental::polygamma` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.polygamma.md) |
| 31 | `at::remainder` | `paddle::experimental::remainder` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.remainder.md) |
| 32 | `at::renorm` | `paddle::experimental::renorm` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.renorm.md) |
| 33 | `at::softplus` | `paddle::experimental::softplus` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.softplus.md) |
| 34 | `at::stack` | `paddle::experimental::stack` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.stack.md) |
| 35 | `at::tril` | `paddle::experimental::tril` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.tril.md) |
| 36 | `at::triu` | `paddle::experimental::triu` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.triu.md) |
| 37 | `at::unbind` | `paddle::experimental::unbind` | 输入参数类型不一致 | [差异对比](cpp_input_args_type_diff/at.unbind.md) |

### 9. 返回参数类型不一致

**简介：** 此类 API 返回值的类型或结构不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::aminmax` | `paddle::experimental::aminmax` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.aminmax.md) |
| 2 | `at::argsort` | `paddle::experimental::argsort` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.argsort.md) |
| 3 | `at::batch_norm` | `paddle::experimental::batch_norm` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.batch_norm.md) |
| 4 | `at::cummax` | `paddle::experimental::cummax` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.cummax.md) |
| 5 | `at::cummin` | `paddle::experimental::cummin` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.cummin.md) |
| 6 | `at::fractional_max_pool2d` | `paddle::experimental::fractional_max_pool2d` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.fractional_max_pool2d.md) |
| 7 | `at::fractional_max_pool3d` | `paddle::experimental::fractional_max_pool3d` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.fractional_max_pool3d.md) |
| 8 | `at::gru` | `paddle::experimental::gru` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.gru.md) |
| 9 | `at::histogram` | `paddle::experimental::histogram` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.histogram.md) |
| 10 | `at::kthvalue` | `paddle::experimental::kthvalue` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.kthvalue.md) |
| 11 | `at::lstm` | `paddle::experimental::lstm` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.lstm.md) |
| 12 | `at::lu_unpack` | `paddle::experimental::lu_unpack` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.lu_unpack.md) |
| 13 | `at::median` | `paddle::experimental::median` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.median.md) |
| 14 | `at::mode` | `paddle::experimental::mode` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.mode.md) |
| 15 | `at::nanmedian` | `paddle::experimental::nanmedian` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.nanmedian.md) |
| 16 | `at::nll_loss` | `paddle::experimental::nll_loss` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.nll_loss.md) |
| 17 | `at::norm` | `paddle::experimental::norm` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.norm.md) |
| 18 | `at::qr` | `paddle::experimental::qr` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.qr.md) |
| 19 | `at::rms_norm` | `paddle::experimental::rms_norm` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.rms_norm.md) |
| 20 | `at::slogdet` | `paddle::experimental::slogdet` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.slogdet.md) |
| 21 | `at::svd` | `paddle::experimental::svd` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.svd.md) |
| 22 | `at::topk` | `paddle::experimental::topk` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.topk.md) |
| 23 | `at::triangular_solve` | `paddle::experimental::triangular_solve` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.triangular_solve.md) |
| 24 | `at::unique_consecutive` | `paddle::experimental::unique_consecutive` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.unique_consecutive.md) |
| 25 | `at::where` | `paddle::experimental::where` | 返回参数类型不一致 | [差异对比](cpp_output_args_type_diff/at.where.md) |
| 26 | `at::einsum` | `paddle::experimental::einsum` | 返回参数类型不一致 | Paddle 返回 tuple，PyTorch 返回 Tensor |

### 10. 组合替代实现

**简介：** 此类功能在 Paddle 中没有直接对应的单一 API，需要通过多个 Paddle API 组合来实现。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| - | - | - | - | 暂无 |

### 11. API 别名

**简介：** 此类 PyTorch API 在 Paddle 中有功能一致的实现，但 API 名称不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::_conj` | `paddle::experimental::conj` | API 别名 | [差异对比](cpp_api_alias_diff/at._conj.md) |
| 2 | `at::_fft_c2c` | `paddle::experimental::fft_c2c` | API 别名 | [差异对比](cpp_api_alias_diff/at._fft_c2c.md) |
| 3 | `at::_fft_c2r` | `paddle::experimental::fft_c2r` | API 别名 | [差异对比](cpp_api_alias_diff/at._fft_c2r.md) |
| 4 | `at::_fft_r2c` | `paddle::experimental::fft_r2c` | API 别名 | [差异对比](cpp_api_alias_diff/at._fft_r2c.md) |
| 5 | `at::_log_softmax` | `paddle::experimental::log_softmax` | API 别名 | [差异对比](cpp_api_alias_diff/at._log_softmax.md) |
| 6 | `at::_logcumsumexp` | `paddle::experimental::logcumsumexp` | API 别名 | [差异对比](cpp_api_alias_diff/at._logcumsumexp.md) |
| 7 | `at::_softmax` | `paddle::experimental::softmax` | API 别名 | [差异对比](cpp_api_alias_diff/at._softmax.md) |
| 8 | `at::_stack` | `paddle::experimental::stack` | API 别名 | [差异对比](cpp_api_alias_diff/at._stack.md) |
| 9 | `at::_standard_gamma` | `paddle::experimental::standard_gamma` | API 别名 | [差异对比](cpp_api_alias_diff/at._standard_gamma.md) |
| 10 | `at::conv_transpose2d` | `paddle::experimental::conv2d_transpose` | API 别名 | [差异对比](cpp_api_alias_diff/at.conv_transpose2d.md) |
| 11 | `at::conv_transpose3d` | `paddle::experimental::conv3d_transpose` | API 别名 | [差异对比](cpp_api_alias_diff/at.conv_transpose3d.md) |
| 12 | `at::grid_sampler` | `paddle::experimental::grid_sample` | API 别名 | [差异对比](cpp_api_alias_diff/at.grid_sampler.md) |
| 13 | `at::log_sigmoid` | `paddle::experimental::logsigmoid` | API 别名 | [差异对比](cpp_api_alias_diff/at.log_sigmoid.md) |

### 12. 语义差异


| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::uniform` | `paddle::experimental::uniform` | 语义差异 | [差异对比](cpp_semantic_mismatch/at.uniform.md) |

### 13. 功能缺失

**简介：** 此类 PyTorch C++ API 在 Paddle 中暂时没有等效实现。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::_adaptive_avg_pool2d` | - | 功能缺失 | 自适应平均池化内部实现 |
| 2 | `at::_adaptive_avg_pool3d` | - | 功能缺失 | 自适应平均池化内部实现 |
| 3 | `at::_add_batch_dim` | - | 功能缺失 | PyTorch vmap内部批量维度操作 |
| 4 | `at::_add_relu` | - | 功能缺失 | PyTorch内部实现API |
| 5 | `at::_addmm_activation` | - | 功能缺失 | addmm+activation融合算子 |
| 6 | `at::_amp_foreach_non_finite_check_and_unscale` | - | 功能缺失 | PyTorch AMP自动混合精度内部API |
| 7 | `at::_amp_update_scale` | - | 功能缺失 | PyTorch AMP自动混合精度内部API |
| 8 | `at::_assert_async` | - | 功能缺失 | PyTorch内部断言检查API |
| 9 | `at::_assert_scalar` | - | 功能缺失 | PyTorch内部断言检查API |
| 10 | `at::_autocast_to_full_precision` | - | 功能缺失 | PyTorch autocast自动类型转换内部API |
| 11 | `at::_autocast_to_reduced_precision` | - | 功能缺失 | PyTorch autocast自动类型转换内部API |
| 12 | `at::_batch_norm_impl_index` | - | 功能缺失 | PyTorch内部实现辅助函数 |
| 13 | `at::_batch_norm_no_update` | - | 功能缺失 | PyTorch内部batch_norm实现入口 |
| 14 | `at::_batch_norm_with_update` | - | 功能缺失 | PyTorch内部batch_norm实现入口 |
| 15 | `at::_cast_Byte` | - | 功能缺失 | PyTorch内部类型转换辅助 |
| 16 | `at::_cast_Char` | - | 功能缺失 | PyTorch内部类型转换辅助 |
| 17 | `at::_cast_Double` | - | 功能缺失 | PyTorch内部类型转换辅助 |
| 18 | `at::_cast_Float` | - | 功能缺失 | PyTorch内部类型转换辅助 |
| 19 | `at::_cast_Half` | - | 功能缺失 | PyTorch内部类型转换辅助 |
| 20 | `at::_cast_Int` | - | 功能缺失 | PyTorch内部类型转换辅助 |
| 21 | `at::_cast_Long` | - | 功能缺失 | PyTorch内部类型转换辅助 |
| 22 | `at::_cast_Short` | - | 功能缺失 | PyTorch内部类型转换辅助 |
| 23 | `at::_cholesky_solve_helper` | - | 功能缺失 | PyTorch内部实现辅助函数 |
| 24 | `at::_choose_qparams_per_tensor` | - | 功能缺失 | 量化参数选择 |
| 25 | `at::_chunk_cat` | - | 功能缺失 | chunk+concat组合操作 |
| 26 | `at::_coalesce` | - | 功能缺失 | PyTorch稀疏张量内部操作 |
| 27 | `at::_coalesced` | - | 功能缺失 | PyTorch稀疏张量内部操作 |
| 28 | `at::_compute_linear_combination` | - | 功能缺失 | 线性组合计算 |
| 29 | `at::_conj_copy` | - | 功能缺失 | 共轭操作内部变体 |
| 30 | `at::_conj_physical` | - | 功能缺失 | 共轭操作内部变体 |
| 31 | `at::_conv_depthwise2d` | - | 功能缺失 | 深度可分离卷积内部实现 |
| 32 | `at::_convert_indices_from_coo_to_csr` | - | 功能缺失 | 稀疏张量索引格式转换 |
| 33 | `at::_convert_indices_from_csr_to_coo` | - | 功能缺失 | 稀疏张量索引格式转换 |
| 34 | `at::_convert_weight_to_int4pack` | - | 功能缺失 | INT4/INT8量化权重操作 |
| 35 | `at::_convolution` | - | 功能缺失 | PyTorch卷积内部实现入口 |
| 36 | `at::_convolution_mode` | - | 功能缺失 | PyTorch卷积内部实现入口 |
| 37 | `at::_copy_from` | - | 功能缺失 | PyTorch内部拷贝实现变体 |
| 38 | `at::_copy_from_and_resize` | - | 功能缺失 | PyTorch内部拷贝实现变体 |
| 39 | `at::_cslt_compress` | - | 功能缺失 | cuSPARSELt压缩 |
| 40 | `at::_ctc_loss` | - | 功能缺失 | CTC Loss内部实现 |
| 41 | `at::_cudnn_ctc_loss` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 42 | `at::_cudnn_init_dropout_state` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 43 | `at::_cudnn_rnn` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 44 | `at::_cudnn_rnn_flatten_weight` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 45 | `at::_cufft_clear_plan_cache` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 46 | `at::_cufft_get_plan_cache_max_size` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 47 | `at::_cufft_get_plan_cache_size` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 48 | `at::_cufft_set_plan_cache_max_size` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 49 | `at::_cummax_helper` | - | 功能缺失 | PyTorch内部实现辅助函数 |
| 50 | `at::_cummin_helper` | - | 功能缺失 | PyTorch内部实现辅助函数 |
| 51 | `at::_debug_has_internal_overlap` | - | 功能缺失 | PyTorch调试工具API |
| 52 | `at::_dimI` | - | 功能缺失 | PyTorch稀疏张量内部操作 |
| 53 | `at::_dimV` | - | 功能缺失 | PyTorch稀疏张量内部操作 |
| 54 | `at::_dim_arange` | - | 功能缺失 | 维度arange辅助 |
| 55 | `at::_dirichlet_grad` | - | 功能缺失 | Dirichlet分布梯度计算 |
| 56 | `at::_dyn_quant_matmul_4bit` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 57 | `at::_dyn_quant_pack_4bit_weight` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 58 | `at::_efficientzerotensor` | - | 功能缺失 | PyTorch零张量优化内部API |
| 59 | `at::_embedding_bag` | - | 功能缺失 | Embedding Bag内部实现 |
| 60 | `at::_euclidean_dist` | - | 功能缺失 | 欧氏距离计算 |
| 61 | `at::_fake_quantize_learnable_per_channel_affine` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 62 | `at::_fake_quantize_learnable_per_tensor_affine` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 63 | `at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 64 | `at::_fill_mem_eff_dropout_mask` | - | 功能缺失 | 内存高效dropout掩码填充 |
| 65 | `at::_foobar` | - | 功能缺失 | PyTorch内部测试占位符 |
| 66 | `at::_foreach_abs` | - | 功能缺失 | PyTorch foreach批量abs操作，可用循环调用基础API替代 |
| 67 | `at::_foreach_acos` | - | 功能缺失 | PyTorch foreach批量acos操作，可用循环调用基础API替代 |
| 68 | `at::_foreach_add` | - | 功能缺失 | PyTorch foreach批量add操作，可用循环调用基础API替代 |
| 69 | `at::_foreach_addcdiv` | - | 功能缺失 | PyTorch foreach批量addcdiv操作，可用循环调用基础API替代 |
| 70 | `at::_foreach_addcmul` | - | 功能缺失 | PyTorch foreach批量addcmul操作，可用循环调用基础API替代 |
| 71 | `at::_foreach_asin` | - | 功能缺失 | PyTorch foreach批量asin操作，可用循环调用基础API替代 |
| 72 | `at::_foreach_atan` | - | 功能缺失 | PyTorch foreach批量atan操作，可用循环调用基础API替代 |
| 73 | `at::_foreach_ceil` | - | 功能缺失 | PyTorch foreach批量ceil操作，可用循环调用基础API替代 |
| 74 | `at::_foreach_clamp_max` | - | 功能缺失 | PyTorch foreach批量clamp_max操作，可用循环调用基础API替代 |
| 75 | `at::_foreach_clamp_min` | - | 功能缺失 | PyTorch foreach批量clamp_min操作，可用循环调用基础API替代 |
| 76 | `at::_foreach_copy` | - | 功能缺失 | PyTorch foreach批量copy操作，可用循环调用基础API替代 |
| 77 | `at::_foreach_cos` | - | 功能缺失 | PyTorch foreach批量cos操作，可用循环调用基础API替代 |
| 78 | `at::_foreach_cosh` | - | 功能缺失 | PyTorch foreach批量cosh操作，可用循环调用基础API替代 |
| 79 | `at::_foreach_div` | - | 功能缺失 | PyTorch foreach批量div操作，可用循环调用基础API替代 |
| 80 | `at::_foreach_erf` | - | 功能缺失 | PyTorch foreach批量erf操作，可用循环调用基础API替代 |
| 81 | `at::_foreach_erfc` | - | 功能缺失 | PyTorch foreach批量erfc操作，可用循环调用基础API替代 |
| 82 | `at::_foreach_exp` | - | 功能缺失 | PyTorch foreach批量exp操作，可用循环调用基础API替代 |
| 83 | `at::_foreach_expm1` | - | 功能缺失 | PyTorch foreach批量expm1操作，可用循环调用基础API替代 |
| 84 | `at::_foreach_floor` | - | 功能缺失 | PyTorch foreach批量floor操作，可用循环调用基础API替代 |
| 85 | `at::_foreach_frac` | - | 功能缺失 | PyTorch foreach批量frac操作，可用循环调用基础API替代 |
| 86 | `at::_foreach_lerp` | - | 功能缺失 | PyTorch foreach批量lerp操作，可用循环调用基础API替代 |
| 87 | `at::_foreach_lgamma` | - | 功能缺失 | PyTorch foreach批量lgamma操作，可用循环调用基础API替代 |
| 88 | `at::_foreach_log` | - | 功能缺失 | PyTorch foreach批量log操作，可用循环调用基础API替代 |
| 89 | `at::_foreach_log10` | - | 功能缺失 | PyTorch foreach批量log10操作，可用循环调用基础API替代 |
| 90 | `at::_foreach_log1p` | - | 功能缺失 | PyTorch foreach批量log1p操作，可用循环调用基础API替代 |
| 91 | `at::_foreach_log2` | - | 功能缺失 | PyTorch foreach批量log2操作，可用循环调用基础API替代 |
| 92 | `at::_foreach_max` | - | 功能缺失 | PyTorch foreach批量max操作，可用循环调用基础API替代 |
| 93 | `at::_foreach_maximum` | - | 功能缺失 | PyTorch foreach批量maximum操作，可用循环调用基础API替代 |
| 94 | `at::_foreach_minimum` | - | 功能缺失 | PyTorch foreach批量minimum操作，可用循环调用基础API替代 |
| 95 | `at::_foreach_mul` | - | 功能缺失 | PyTorch foreach批量mul操作，可用循环调用基础API替代 |
| 96 | `at::_foreach_neg` | - | 功能缺失 | PyTorch foreach批量neg操作，可用循环调用基础API替代 |
| 97 | `at::_foreach_norm` | - | 功能缺失 | PyTorch foreach批量norm操作，可用循环调用基础API替代 |
| 98 | `at::_foreach_pow` | - | 功能缺失 | PyTorch foreach批量pow操作，可用循环调用基础API替代 |
| 99 | `at::_foreach_reciprocal` | - | 功能缺失 | PyTorch foreach批量reciprocal操作，可用循环调用基础API替代 |
| 100 | `at::_foreach_round` | - | 功能缺失 | PyTorch foreach批量round操作，可用循环调用基础API替代 |
| 101 | `at::_foreach_rsqrt` | - | 功能缺失 | PyTorch foreach批量rsqrt操作，可用循环调用基础API替代 |
| 102 | `at::_foreach_sigmoid` | - | 功能缺失 | PyTorch foreach批量sigmoid操作，可用循环调用基础API替代 |
| 103 | `at::_foreach_sign` | - | 功能缺失 | PyTorch foreach批量sign操作，可用循环调用基础API替代 |
| 104 | `at::_foreach_sin` | - | 功能缺失 | PyTorch foreach批量sin操作，可用循环调用基础API替代 |
| 105 | `at::_foreach_sinh` | - | 功能缺失 | PyTorch foreach批量sinh操作，可用循环调用基础API替代 |
| 106 | `at::_foreach_sqrt` | - | 功能缺失 | PyTorch foreach批量sqrt操作，可用循环调用基础API替代 |
| 107 | `at::_foreach_sub` | - | 功能缺失 | PyTorch foreach批量sub操作，可用循环调用基础API替代 |
| 108 | `at::_foreach_tan` | - | 功能缺失 | PyTorch foreach批量tan操作，可用循环调用基础API替代 |
| 109 | `at::_foreach_tanh` | - | 功能缺失 | PyTorch foreach批量tanh操作，可用循环调用基础API替代 |
| 110 | `at::_foreach_trunc` | - | 功能缺失 | PyTorch foreach批量trunc操作，可用循环调用基础API替代 |
| 111 | `at::_foreach_zero` | - | 功能缺失 | PyTorch foreach批量zero操作，可用循环调用基础API替代 |
| 112 | `at::_fused_adagrad` | - | 功能缺失 | PyTorch融合算子，Paddle有对应融合实现 |
| 113 | `at::_fused_adam` | - | 功能缺失 | PyTorch融合算子，Paddle有对应融合实现 |
| 114 | `at::_fused_adamw` | - | 功能缺失 | PyTorch融合算子，Paddle有对应融合实现 |
| 115 | `at::_fused_dropout` | - | 功能缺失 | PyTorch融合算子，Paddle有对应融合实现 |
| 116 | `at::_fused_moving_avg_obs_fq_helper` | - | 功能缺失 | PyTorch内部实现辅助函数 |
| 117 | `at::_fused_rms_norm` | - | 功能缺失 | PyTorch融合算子，Paddle有对应融合实现 |
| 118 | `at::_fused_sdp_choice` | - | 功能缺失 | PyTorch SDPA/Flash Attention，Paddle有独立flash_attn实现 |
| 119 | `at::_fused_sgd` | - | 功能缺失 | PyTorch融合算子，Paddle有对应融合实现 |
| 120 | `at::_fw_primal` | - | 功能缺失 | PyTorch前向模式自动微分内部API |
| 121 | `at::_fw_primal_copy` | - | 功能缺失 | PyTorch前向模式自动微分内部API |
| 122 | `at::_grouped_mm` | - | 功能缺失 | 分组矩阵乘法 |
| 123 | `at::_has_compatible_shallow_copy_type` | - | 功能缺失 | PyTorch内部属性检查API |
| 124 | `at::_has_same_storage_numel` | - | 功能缺失 | PyTorch内部属性检查API |
| 125 | `at::_histogramdd_bin_edges` | - | 功能缺失 | 多维直方图 |
| 126 | `at::_histogramdd_from_bin_cts` | - | 功能缺失 | 多维直方图 |
| 127 | `at::_histogramdd_from_bin_tensors` | - | 功能缺失 | 多维直方图 |
| 128 | `at::_index_put_impl` | - | 功能缺失 | 索引赋值内部实现 |
| 129 | `at::_indices` | - | 功能缺失 | PyTorch稀疏张量内部操作 |
| 130 | `at::_indices_copy` | - | 功能缺失 | PyTorch内部实现API |
| 131 | `at::_int_mm` | - | 功能缺失 | PyTorch内部实现API |
| 132 | `at::_is_all_true` | - | 功能缺失 | PyTorch内部状态检查API |
| 133 | `at::_is_any_true` | - | 功能缺失 | PyTorch内部状态检查API |
| 134 | `at::_is_zerotensor` | - | 功能缺失 | PyTorch内部状态检查API |
| 135 | `at::_lazy_clone` | - | 功能缺失 | PyTorch惰性拷贝内部API |
| 136 | `at::_linalg_check_errors` | - | 功能缺失 | PyTorch linalg内部实现入口 |
| 137 | `at::_linalg_det` | - | 功能缺失 | PyTorch linalg内部实现入口 |
| 138 | `at::_linalg_eigh` | - | 功能缺失 | PyTorch linalg内部实现入口 |
| 139 | `at::_linalg_eigvals` | - | 功能缺失 | PyTorch linalg内部实现入口 |
| 140 | `at::_linalg_slogdet` | - | 功能缺失 | PyTorch linalg内部实现入口 |
| 141 | `at::_linalg_solve_ex` | - | 功能缺失 | PyTorch linalg内部实现入口 |
| 142 | `at::_linalg_svd` | - | 功能缺失 | PyTorch linalg内部实现入口 |
| 143 | `at::_lu_with_info` | - | 功能缺失 | LU分解带信息输出 |
| 144 | `at::_make_dep_token` | - | 功能缺失 | 依赖token内部辅助 |
| 145 | `at::_make_dual` | - | 功能缺失 | PyTorch前向模式自动微分内部API |
| 146 | `at::_make_dual_copy` | - | 功能缺失 | PyTorch前向模式自动微分内部API |
| 147 | `at::_masked_scale` | - | 功能缺失 | 掩码缩放 |
| 148 | `at::_masked_softmax` | - | 功能缺失 | 掩码Softmax内部变体 |
| 149 | `at::_mixed_dtypes_linear` | - | 功能缺失 | 混合精度线性层 |
| 150 | `at::_neg_view` | - | 功能缺失 | 负视图内部实现 |
| 151 | `at::_neg_view_copy` | - | 功能缺失 | 负视图内部实现 |
| 152 | `at::_nested_compute_contiguous_strides_offsets` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 153 | `at::_nested_from_padded` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 154 | `at::_nested_from_padded_and_nested_example` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 155 | `at::_nested_from_padded_tensor` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 156 | `at::_nested_get_jagged_dummy` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 157 | `at::_nested_get_lengths` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 158 | `at::_nested_get_max_seqlen` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 159 | `at::_nested_get_min_seqlen` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 160 | `at::_nested_get_offsets` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 161 | `at::_nested_get_ragged_idx` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 162 | `at::_nested_get_values` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 163 | `at::_nested_get_values_copy` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 164 | `at::_nested_tensor_from_mask` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 165 | `at::_nested_tensor_from_mask_left_aligned` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 166 | `at::_nested_tensor_from_tensor_list` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 167 | `at::_nested_tensor_size` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 168 | `at::_nested_tensor_softmax_with_shape` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 169 | `at::_nested_tensor_storage_offsets` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 170 | `at::_nested_tensor_strides` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 171 | `at::_nested_view_from_buffer` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 172 | `at::_nested_view_from_buffer_copy` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 173 | `at::_nested_view_from_jagged` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 174 | `at::_nested_view_from_jagged_copy` | - | 功能缺失 | PyTorch 2.0+嵌套张量特性，Paddle暂不支持 |
| 175 | `at::_nnpack_available` | - | 功能缺失 | NNPACK专用API |
| 176 | `at::_nnpack_spatial_convolution` | - | 功能缺失 | NNPACK专用API |
| 177 | `at::_pack_padded_sequence` | - | 功能缺失 | 变长序列填充/打包 |
| 178 | `at::_pad_circular` | - | 功能缺失 | 循环填充 |
| 179 | `at::_pad_enum` | - | 功能缺失 | 填充模式枚举 |
| 180 | `at::_pad_packed_sequence` | - | 功能缺失 | 变长序列填充/打包 |
| 181 | `at::_pin_memory` | - | 功能缺失 | PyTorch内部内存钉住辅助 |
| 182 | `at::_prelu_kernel` | - | 功能缺失 | PReLU kernel内部入口 |
| 183 | `at::_print` | - | 功能缺失 | PyTorch内部打印调试API |
| 184 | `at::_propagate_xla_data` | - | 功能缺失 | XLA专用内部API |
| 185 | `at::_remove_batch_dim` | - | 功能缺失 | PyTorch vmap内部批量维度操作 |
| 186 | `at::_reshape_alias` | - | 功能缺失 | PyTorch内部reshape实现变体 |
| 187 | `at::_reshape_alias_copy` | - | 功能缺失 | PyTorch内部reshape实现变体 |
| 188 | `at::_reshape_copy` | - | 功能缺失 | PyTorch内部reshape实现变体 |
| 189 | `at::_reshape_from_tensor` | - | 功能缺失 | PyTorch内部reshape实现变体 |
| 190 | `at::_resize_output` | - | 功能缺失 | PyTorch内部输出resize辅助 |
| 191 | `at::_rowwise_prune` | - | 功能缺失 | 行级剪枝 |
| 192 | `at::_safe_softmax` | - | 功能缺失 | 安全Softmax内部变体 |
| 193 | `at::_sample_dirichlet` | - | 功能缺失 | Dirichlet分布采样 |
| 194 | `at::_saturate_weight_to_fp16` | - | 功能缺失 | FP16权重饱和转换 |
| 195 | `at::_scaled_dot_product_cudnn_attention` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 196 | `at::_scaled_dot_product_efficient_attention` | - | 功能缺失 | PyTorch SDPA/Flash Attention，Paddle有独立flash_attn实现 |
| 197 | `at::_scaled_dot_product_flash_attention` | - | 功能缺失 | PyTorch SDPA/Flash Attention，Paddle有独立flash_attn实现 |
| 198 | `at::_scaled_dot_product_fused_attention_overrideable` | - | 功能缺失 | PyTorch SDPA/Flash Attention，Paddle有独立flash_attn实现 |
| 199 | `at::_scaled_grouped_mm` | - | 功能缺失 | 分组矩阵乘法 |
| 200 | `at::_scaled_grouped_mm_v2` | - | 功能缺失 | 分组矩阵乘法 |
| 201 | `at::_scaled_mm` | - | 功能缺失 | 缩放矩阵乘法 |
| 202 | `at::_scaled_mm_v2` | - | 功能缺失 | 缩放矩阵乘法 |
| 203 | `at::_shape_as_tensor` | - | 功能缺失 | PyTorch内部shape转tensor辅助 |
| 204 | `at::_sobol_engine_draw` | - | 功能缺失 | Sobol序列拟随机数生成 |
| 205 | `at::_sobol_engine_ff` | - | 功能缺失 | Sobol序列拟随机数生成 |
| 206 | `at::_sobol_engine_initialize_state` | - | 功能缺失 | Sobol序列拟随机数生成 |
| 207 | `at::_sobol_engine_scramble` | - | 功能缺失 | Sobol序列拟随机数生成 |
| 208 | `at::_spdiags` | - | 功能缺失 | 稀疏对角矩阵 |
| 209 | `at::_spsolve` | - | 功能缺失 | 稀疏矩阵求解 |
| 210 | `at::_standard_gamma_grad` | - | 功能缺失 | Standard Gamma梯度计算 |
| 211 | `at::_test_ambiguous_defaults` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 212 | `at::_test_check_tensor` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 213 | `at::_test_functorch_fallback` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 214 | `at::_test_optional_filled_intlist` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 215 | `at::_test_optional_floatlist` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 216 | `at::_test_optional_intlist` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 217 | `at::_test_parallel_materialize` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 218 | `at::_test_serialization_subcmul` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 219 | `at::_test_string_default` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 220 | `at::_test_warn_in_autograd` | - | 功能缺失 | PyTorch内部测试API，无需映射 |
| 221 | `at::_thnn_fused_gru_cell` | - | 功能缺失 | PyTorch THNN融合实现 |
| 222 | `at::_thnn_fused_lstm_cell` | - | 功能缺失 | PyTorch THNN融合实现 |
| 223 | `at::_to_copy` | - | 功能缺失 | PyTorch内部拷贝实现变体 |
| 224 | `at::_to_dense` | - | 功能缺失 | PyTorch内部实现API |
| 225 | `at::_transform_bias_rescale_qkv` | - | 功能缺失 | Transformer QKV偏置缩放 |
| 226 | `at::_transformer_encoder_layer_fwd` | - | 功能缺失 | Transformer编码器层前向内部实现 |
| 227 | `at::_trilinear` | - | 功能缺失 | 三线插值 |
| 228 | `at::_triton_multi_head_attention` | - | 功能缺失 | OpenAI Triton集成专用API |
| 229 | `at::_triton_scaled_dot_attention` | - | 功能缺失 | OpenAI Triton集成专用API |
| 230 | `at::_unique2` | - | 功能缺失 | 去重内部变体 |
| 231 | `at::_unpack_dual` | - | 功能缺失 | PyTorch前向模式自动微分内部API |
| 232 | `at::_unsafe_index` | - | 功能缺失 | PyTorch内部非安全检查变体 |
| 233 | `at::_unsafe_index_put` | - | 功能缺失 | PyTorch内部非安全检查变体 |
| 234 | `at::_unsafe_masked_index` | - | 功能缺失 | PyTorch内部非安全检查变体 |
| 235 | `at::_unsafe_masked_index_put_accumulate` | - | 功能缺失 | PyTorch内部非安全检查变体 |
| 236 | `at::_unsafe_view` | - | 功能缺失 | PyTorch内部非安全检查变体 |
| 237 | `at::_upsample_bicubic2d_aa` | - | 功能缺失 | 上采样内部实现变体 |
| 238 | `at::_upsample_bilinear2d_aa` | - | 功能缺失 | 上采样内部实现变体 |
| 239 | `at::_upsample_nearest_exact1d` | - | 功能缺失 | 上采样内部实现变体 |
| 240 | `at::_upsample_nearest_exact2d` | - | 功能缺失 | 上采样内部实现变体 |
| 241 | `at::_upsample_nearest_exact3d` | - | 功能缺失 | 上采样内部实现变体 |
| 242 | `at::_use_cudnn_ctc_loss` | - | 功能缺失 | PyTorch内部实现辅助函数 |
| 243 | `at::_use_cudnn_rnn_flatten_weight` | - | 功能缺失 | PyTorch内部实现辅助函数 |
| 244 | `at::_values_copy` | - | 功能缺失 | PyTorch内部实现API |
| 245 | `at::_version` | - | 功能缺失 | PyTorch内部版本追踪API |
| 246 | `at::_weight_int4pack_mm` | - | 功能缺失 | INT4/INT8量化权重操作 |
| 247 | `at::_weight_int4pack_mm_with_scales_and_zeros` | - | 功能缺失 | INT4/INT8量化权重操作 |
| 248 | `at::_weight_int8pack_mm` | - | 功能缺失 | INT4/INT8量化权重操作 |
| 249 | `at::_weight_norm` | - | 功能缺失 | 权重归一化 |
| 250 | `at::_weight_norm_interface` | - | 功能缺失 | 权重归一化 |
| 251 | `at::_wrapped_linear_prepack` | - | 功能缺失 | PyTorch内部实现API |
| 252 | `at::absolute` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 253 | `at::adaptive_avg_pool1d` | - | 功能缺失 | 池化操作变体 |
| 254 | `at::adaptive_avg_pool2d` | - | 功能缺失 | 池化操作变体 |
| 255 | `at::adaptive_avg_pool3d` | - | 功能缺失 | 池化操作变体 |
| 256 | `at::adaptive_max_pool1d` | - | 功能缺失 | 池化操作变体 |
| 257 | `at::adaptive_max_pool2d` | - | 功能缺失 | 池化操作变体 |
| 258 | `at::adaptive_max_pool3d` | - | 功能缺失 | 池化操作变体 |
| 259 | `at::addbmm` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 260 | `at::addcdiv` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 261 | `at::addcmul` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 262 | `at::addmv` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 263 | `at::addr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 264 | `at::adjoint` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 265 | `at::affine_grid_generator` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 266 | `at::alias` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 267 | `at::alias_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 268 | `at::align_as` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 269 | `at::align_tensors` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 270 | `at::align_to` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 271 | `at::alpha_dropout` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 272 | `at::and` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 273 | `at::arccos` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 274 | `at::arccosh` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 275 | `at::arcsin` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 276 | `at::arcsinh` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 277 | `at::arctan` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 278 | `at::arctan2` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 279 | `at::arctanh` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 280 | `at::argwhere` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 281 | `at::as_strided_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 282 | `at::as_strided_scatter` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 283 | `at::atleast_1d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 284 | `at::atleast_2d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 285 | `at::atleast_3d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 286 | `at::avg_pool1d` | - | 功能缺失 | 池化操作变体 |
| 287 | `at::avg_pool2d` | - | 功能缺失 | 池化操作变体 |
| 288 | `at::avg_pool3d` | - | 功能缺失 | 池化操作变体 |
| 289 | `at::bartlett_window` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 290 | `at::batch_norm_elemt` | - | 功能缺失 | 归一化操作变体 |
| 291 | `at::batch_norm_gather_stats` | - | 功能缺失 | 归一化操作变体 |
| 292 | `at::batch_norm_gather_stats_with_counts` | - | 功能缺失 | 归一化操作变体 |
| 293 | `at::batch_norm_stats` | - | 功能缺失 | 归一化操作变体 |
| 294 | `at::batch_norm_update_stats` | - | 功能缺失 | 归一化操作变体 |
| 295 | `at::binary_cross_entropy` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 296 | `at::binary_cross_entropy_with_logits` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 297 | `at::blackman_window` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 298 | `at::block_diag` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 299 | `at::broadcast_to` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 300 | `at::bucketize` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 301 | `at::can_cast` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 302 | `at::cartesian_prod` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 303 | `at::cauchy` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 304 | `at::ccol_indices` | - | 功能缺失 | CSR格式索引访问器 |
| 305 | `at::ccol_indices_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 306 | `at::cdist` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 307 | `at::chain_matmul` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 308 | `at::chalf` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 309 | `at::cholesky_inverse` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 310 | `at::choose_qparams_optimized` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 311 | `at::clamp_max` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 312 | `at::clamp_min` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 313 | `at::clone` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 314 | `at::col2im` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 315 | `at::col_indices` | - | 功能缺失 | CSR格式索引访问器 |
| 316 | `at::col_indices_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 317 | `at::column_stack` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 318 | `at::combinations` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 319 | `at::concatenate` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 320 | `at::conj_physical` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 321 | `at::constant_pad_nd` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 322 | `at::contiguous` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 323 | `at::conv1d` | - | 功能缺失 | 卷积操作变体 |
| 324 | `at::conv_depthwise3d` | - | 功能缺失 | 卷积操作变体 |
| 325 | `at::conv_tbc` | - | 功能缺失 | 卷积操作变体 |
| 326 | `at::conv_transpose1d` | - | 功能缺失 | 卷积操作变体 |
| 327 | `at::convolution` | - | 功能缺失 | 卷积操作变体 |
| 328 | `at::convolution_overrideable` | - | 功能缺失 | 卷积操作变体 |
| 329 | `at::copy` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 330 | `at::corrcoef` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 331 | `at::cosine_embedding_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 332 | `at::cosine_similarity` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 333 | `at::count_nonzero` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 334 | `at::cov` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 335 | `at::cross_entropy_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 336 | `at::crow_indices` | - | 功能缺失 | CSR格式索引访问器 |
| 337 | `at::crow_indices_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 338 | `at::ctc_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 339 | `at::cudnn_affine_grid_generator` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 340 | `at::cudnn_batch_norm` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 341 | `at::cudnn_convolution` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 342 | `at::cudnn_convolution_add_relu` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 343 | `at::cudnn_convolution_relu` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 344 | `at::cudnn_convolution_transpose` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 345 | `at::cudnn_grid_sampler` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 346 | `at::cudnn_is_acceptable` | - | 功能缺失 | NVIDIA cuDNN/cuFFT底层专用API，Paddle有独立实现 |
| 347 | `at::cumulative_trapezoid` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 348 | `at::deg2rad` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 349 | `at::dense_dim` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 350 | `at::dequantize` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 351 | `at::detach_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 352 | `at::diagflat` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 353 | `at::diagonal_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 354 | `at::diagonal_scatter` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 355 | `at::diff` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 356 | `at::div` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 357 | `at::dstack` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 359 | `at::embedding_bag` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 360 | `at::embedding_renorm` | - | 功能缺失 | 归一化操作变体 |
| 361 | `at::empty_permuted` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 362 | `at::eq` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 363 | `at::erfc` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 364 | `at::exp2` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 365 | `at::expand_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 366 | `at::exponential` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 367 | `at::fake_quantize_per_channel_affine` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 368 | `at::fake_quantize_per_channel_affine_cachemask` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 369 | `at::fake_quantize_per_tensor_affine` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 370 | `at::fake_quantize_per_tensor_affine_cachemask` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 371 | `at::fbgemm_linear_fp16_weight` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 372 | `at::fbgemm_linear_fp16_weight_fp32_activation` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 373 | `at::fbgemm_linear_int8_weight` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 374 | `at::fbgemm_linear_int8_weight_fp32_activation` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 375 | `at::fbgemm_linear_quantize_weight` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 376 | `at::fbgemm_pack_gemm_matrix_fp16` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 377 | `at::feature_alpha_dropout` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 378 | `at::feature_dropout` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 379 | `at::fft_fft` | - | 功能缺失 | FFT操作变体 |
| 380 | `at::fft_fft2` | - | 功能缺失 | FFT操作变体 |
| 381 | `at::fft_fftfreq` | - | 功能缺失 | FFT操作变体 |
| 382 | `at::fft_fftn` | - | 功能缺失 | FFT操作变体 |
| 383 | `at::fft_fftshift` | - | 功能缺失 | FFT操作变体 |
| 384 | `at::fft_hfft` | - | 功能缺失 | FFT操作变体 |
| 385 | `at::fft_hfft2` | - | 功能缺失 | FFT操作变体 |
| 386 | `at::fft_hfftn` | - | 功能缺失 | FFT操作变体 |
| 387 | `at::fft_ifft` | - | 功能缺失 | FFT操作变体 |
| 388 | `at::fft_ifft2` | - | 功能缺失 | FFT操作变体 |
| 389 | `at::fft_ifftn` | - | 功能缺失 | FFT操作变体 |
| 390 | `at::fft_ifftshift` | - | 功能缺失 | FFT操作变体 |
| 391 | `at::fft_ihfft` | - | 功能缺失 | FFT操作变体 |
| 392 | `at::fft_ihfft2` | - | 功能缺失 | FFT操作变体 |
| 393 | `at::fft_ihfftn` | - | 功能缺失 | FFT操作变体 |
| 394 | `at::fft_irfft` | - | 功能缺失 | FFT操作变体 |
| 395 | `at::fft_irfft2` | - | 功能缺失 | FFT操作变体 |
| 396 | `at::fft_irfftn` | - | 功能缺失 | FFT操作变体 |
| 397 | `at::fft_rfft` | - | 功能缺失 | FFT操作变体 |
| 398 | `at::fft_rfft2` | - | 功能缺失 | FFT操作变体 |
| 399 | `at::fft_rfftfreq` | - | 功能缺失 | FFT操作变体 |
| 400 | `at::fft_rfftn` | - | 功能缺失 | FFT操作变体 |
| 401 | `at::fix` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 402 | `at::flatten_dense_tensors` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 403 | `at::fliplr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 404 | `at::flipud` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 405 | `at::float_power` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 406 | `at::fmod` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 407 | `at::frac` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 408 | `at::frexp` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 409 | `at::from_file` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 410 | `at::fused_moving_avg_obs_fake_quant` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 411 | `at::gcd` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 412 | `at::ge` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 413 | `at::geometric` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 414 | `at::geqrf` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 415 | `at::ger` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 416 | `at::glu` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 417 | `at::glu_jvp` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 418 | `at::gradient` | - | 功能缺失 | 梯度计算 |
| 419 | `at::greater` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 420 | `at::grid_sampler_2d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 421 | `at::grid_sampler_3d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 422 | `at::gru_cell` | - | 功能缺失 | 循环神经网络操作 |
| 423 | `at::gt` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 424 | `at::hamming_window` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 425 | `at::hann_window` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 426 | `at::hash_tensor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 427 | `at::hinge_embedding_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 428 | `at::histc` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 429 | `at::histogramdd` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 430 | `at::hspmm` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 431 | `at::hstack` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 432 | `at::hypot` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 433 | `at::igamma` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 434 | `at::igammac` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 435 | `at::im2col` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 436 | `at::index_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 437 | `at::index_reduce` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 438 | `at::indices` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 439 | `at::indices_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 440 | `at::inner` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 441 | `at::int_repr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 442 | `at::is_complex` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 443 | `at::is_conj` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 444 | `at::is_distributed` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 445 | `at::is_floating_point` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 446 | `at::is_inference` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 447 | `at::is_leaf` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 448 | `at::is_neg` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 449 | `at::is_nonzero` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 450 | `at::is_pinned` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 451 | `at::is_same_size` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 452 | `at::is_set_to` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 453 | `at::is_signed` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 454 | `at::isin` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 455 | `at::isneginf` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 456 | `at::isposinf` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 457 | `at::isreal` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 458 | `at::istft` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 459 | `at::kaiser_window` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 460 | `at::kl_div` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 461 | `at::l1_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 462 | `at::lcm` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 463 | `at::ldexp` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 464 | `at::le` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 465 | `at::less` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 466 | `at::lift` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 467 | `at::lift_fresh` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 468 | `at::lift_fresh_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 469 | `at::linalg_cholesky` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 470 | `at::linalg_cholesky_ex` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 471 | `at::linalg_cond` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 472 | `at::linalg_cross` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 473 | `at::linalg_det` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 474 | `at::linalg_diagonal` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 475 | `at::linalg_eig` | - | 功能缺失 | 特征值分解 |
| 476 | `at::linalg_eigh` | - | 功能缺失 | 特征值分解 |
| 477 | `at::linalg_eigvals` | - | 功能缺失 | 特征值分解 |
| 478 | `at::linalg_eigvalsh` | - | 功能缺失 | 特征值分解 |
| 479 | `at::linalg_householder_product` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 480 | `at::linalg_inv` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 481 | `at::linalg_inv_ex` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 482 | `at::linalg_ldl_factor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 483 | `at::linalg_ldl_factor_ex` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 484 | `at::linalg_ldl_solve` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 485 | `at::linalg_lstsq` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 486 | `at::linalg_lu` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 487 | `at::linalg_lu_factor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 488 | `at::linalg_lu_factor_ex` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 489 | `at::linalg_lu_solve` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 490 | `at::linalg_matmul` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 491 | `at::linalg_matrix_exp` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 492 | `at::linalg_matrix_norm` | - | 功能缺失 | 归一化操作变体 |
| 493 | `at::linalg_matrix_power` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 494 | `at::linalg_matrix_rank` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 495 | `at::linalg_multi_dot` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 496 | `at::linalg_norm` | - | 功能缺失 | 归一化操作变体 |
| 497 | `at::linalg_pinv` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 498 | `at::linalg_qr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 499 | `at::linalg_slogdet` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 500 | `at::linalg_solve` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 501 | `at::linalg_solve_ex` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 502 | `at::linalg_solve_triangular` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 503 | `at::linalg_svd` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 504 | `at::linalg_svdvals` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 505 | `at::linalg_tensorinv` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 506 | `at::linalg_tensorsolve` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 507 | `at::linalg_vander` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 508 | `at::linalg_vecdot` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 509 | `at::linalg_vector_norm` | - | 功能缺失 | 归一化操作变体 |
| 510 | `at::linear` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 511 | `at::log_normal` | - | 功能缺失 | 归一化操作变体 |
| 512 | `at::logaddexp` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 513 | `at::logaddexp2` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 514 | `at::logdet` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 515 | `at::lshift` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 516 | `at::lstm_cell` | - | 功能缺失 | 循环神经网络操作 |
| 517 | `at::lt` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 518 | `at::mH` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 519 | `at::mT` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 520 | `at::margin_ranking_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 521 | `at::matrix_H` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 522 | `at::matrix_exp` | - | 功能缺失 | 矩阵指数/幂运算 |
| 523 | `at::max_pool1d` | - | 功能缺失 | 池化操作变体 |
| 524 | `at::max_pool1d_with_indices` | - | 功能缺失 | 池化操作变体 |
| 525 | `at::max_pool2d` | - | 功能缺失 | 池化操作变体 |
| 526 | `at::max_pool3d` | - | 功能缺失 | 池化操作变体 |
| 527 | `at::max_unpool2d` | - | 功能缺失 | 池化操作变体 |
| 528 | `at::max_unpool3d` | - | 功能缺失 | 池化操作变体 |
| 529 | `at::miopen_batch_norm` | - | 功能缺失 | 归一化操作变体 |
| 530 | `at::miopen_convolution` | - | 功能缺失 | 卷积操作变体 |
| 531 | `at::miopen_convolution_add_relu` | - | 功能缺失 | 卷积操作变体 |
| 532 | `at::miopen_convolution_relu` | - | 功能缺失 | 卷积操作变体 |
| 533 | `at::miopen_convolution_transpose` | - | 功能缺失 | 卷积操作变体 |
| 534 | `at::miopen_depthwise_convolution` | - | 功能缺失 | 卷积操作变体 |
| 535 | `at::miopen_rnn` | - | 功能缺失 | 循环神经网络操作 |
| 536 | `at::mkldnn_adaptive_avg_pool2d` | - | 功能缺失 | Intel MKL-DNN专用API |
| 537 | `at::mkldnn_convolution` | - | 功能缺失 | Intel MKL-DNN专用API |
| 538 | `at::mkldnn_linear` | - | 功能缺失 | Intel MKL-DNN专用API |
| 539 | `at::mkldnn_max_pool2d` | - | 功能缺失 | Intel MKL-DNN专用API |
| 540 | `at::mkldnn_max_pool3d` | - | 功能缺失 | Intel MKL-DNN专用API |
| 541 | `at::mkldnn_reorder_conv2d_weight` | - | 功能缺失 | Intel MKL-DNN专用API |
| 542 | `at::mkldnn_reorder_conv3d_weight` | - | 功能缺失 | Intel MKL-DNN专用API |
| 543 | `at::mkldnn_rnn_layer` | - | 功能缺失 | Intel MKL-DNN专用API |
| 544 | `at::mm` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 545 | `at::moveaxis` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 546 | `at::movedim` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 547 | `at::mse_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 548 | `at::msort` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 549 | `at::mul` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 550 | `at::multi_margin_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 551 | `at::multilabel_margin_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 552 | `at::mvlgamma` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 553 | `at::nan_to_num` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 554 | `at::nanmean` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 555 | `at::nanquantile` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 556 | `at::native_batch_norm` | - | 功能缺失 | 归一化操作变体 |
| 557 | `at::native_channel_shuffle` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 558 | `at::native_dropout` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 559 | `at::native_group_norm` | - | 功能缺失 | 归一化操作变体 |
| 560 | `at::native_layer_norm` | - | 功能缺失 | 归一化操作变体 |
| 561 | `at::native_norm` | - | 功能缺失 | 归一化操作变体 |
| 562 | `at::ne` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 563 | `at::neg` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 564 | `at::negative` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 565 | `at::nested_to_padded_tensor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 566 | `at::new_empty_strided` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 567 | `at::nll_loss2d` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 568 | `at::nll_loss_nd` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 569 | `at::nonzero_numpy` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 570 | `at::nonzero_static` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 571 | `at::norm_except_dim` | - | 功能缺失 | 归一化操作变体 |
| 572 | `at::normal` | - | 功能缺失 | 归一化操作变体 |
| 573 | `at::nuclear_norm` | - | 功能缺失 | 归一化操作变体 |
| 574 | `at::numpy_T` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 575 | `at::or` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 576 | `at::orgqr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 577 | `at::ormqr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 578 | `at::outer` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 579 | `at::output_nr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 580 | `at::pad_sequence` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 581 | `at::pairwise_distance` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 582 | `at::pdist` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 583 | `at::permute_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 584 | `at::pin_memory` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 585 | `at::pinverse` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 586 | `at::poisson_nll_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 587 | `at::polar` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 588 | `at::positive` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 589 | `at::promote_types` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 590 | `at::put` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 591 | `at::q_per_channel_axis` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 592 | `at::q_per_channel_scales` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 593 | `at::q_per_channel_zero_points` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 594 | `at::q_scale` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 595 | `at::q_zero_point` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 596 | `at::qscheme` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 597 | `at::quantile` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 598 | `at::quantize_per_channel` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 599 | `at::quantize_per_tensor` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 600 | `at::quantize_per_tensor_dynamic` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 601 | `at::quantized_batch_norm` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 602 | `at::quantized_gru_cell` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 603 | `at::quantized_lstm_cell` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 604 | `at::quantized_max_pool1d` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 605 | `at::quantized_max_pool2d` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 606 | `at::quantized_max_pool3d` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 607 | `at::quantized_rnn_relu_cell` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 608 | `at::quantized_rnn_tanh_cell` | - | 功能缺失 | 量化相关API，Paddle量化框架设计不同 |
| 609 | `at::rad2deg` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 610 | `at::rand` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 611 | `at::rand_like` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 612 | `at::randint_like` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 613 | `at::randn` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 614 | `at::randn_like` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 615 | `at::ravel` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 616 | `at::refine_names` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 617 | `at::reflection_pad1d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 618 | `at::reflection_pad2d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 619 | `at::reflection_pad3d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 620 | `at::repeat` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 621 | `at::replication_pad1d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 622 | `at::replication_pad2d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 623 | `at::replication_pad3d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 624 | `at::requires_grad` | - | 功能缺失 | 梯度计算 |
| 625 | `at::reshape_as` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 626 | `at::resize_as` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 627 | `at::resolve_conj` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 628 | `at::resolve_neg` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 629 | `at::result_type` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 630 | `at::retain_grad` | - | 功能缺失 | 梯度计算 |
| 631 | `at::retains_grad` | - | 功能缺失 | 梯度计算 |
| 632 | `at::rnn_relu` | - | 功能缺失 | 循环神经网络操作 |
| 633 | `at::rnn_relu_cell` | - | 功能缺失 | 循环神经网络操作 |
| 634 | `at::rnn_tanh` | - | 功能缺失 | 循环神经网络操作 |
| 635 | `at::rnn_tanh_cell` | - | 功能缺失 | 循环神经网络操作 |
| 636 | `at::rot90` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 637 | `at::row_indices` | - | 功能缺失 | CSR格式索引访问器 |
| 638 | `at::row_indices_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 639 | `at::row_stack` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 640 | `at::rrelu_with_noise` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 641 | `at::rshift` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 642 | `at::rsub` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 643 | `at::scalar_tensor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 644 | `at::scaled_dot_product_attention` | - | 功能缺失 | PyTorch SDPA/Flash Attention，Paddle有独立flash_attn实现 |
| 645 | `at::scatter_add` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 646 | `at::scatter_reduce` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 647 | `at::segment_reduce` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 648 | `at::select_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 649 | `at::select_scatter` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 650 | `at::set_data` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 651 | `at::sgn` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 652 | `at::signbit` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 653 | `at::sinc` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 654 | `at::size` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 655 | `at::slice_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 656 | `at::slice_inverse` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 657 | `at::slice_scatter` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 658 | `at::slow_conv3d` | - | 功能缺失 | 卷积操作变体 |
| 659 | `at::slow_conv_dilated2d` | - | 功能缺失 | 卷积操作变体 |
| 660 | `at::slow_conv_dilated3d` | - | 功能缺失 | 卷积操作变体 |
| 661 | `at::slow_conv_transpose2d` | - | 功能缺失 | 卷积操作变体 |
| 662 | `at::slow_conv_transpose3d` | - | 功能缺失 | 卷积操作变体 |
| 663 | `at::smm` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 664 | `at::smooth_l1_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 665 | `at::soft_margin_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 666 | `at::sort` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 667 | `at::sparse_bsc_tensor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 668 | `at::sparse_bsr_tensor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 669 | `at::sparse_compressed_tensor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 670 | `at::sparse_csc_tensor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 671 | `at::sparse_dim` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 672 | `at::sparse_mask` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 673 | `at::sparse_resize` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 674 | `at::sparse_resize_and_clear` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 675 | `at::sparse_sampled_addmm` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 676 | `at::special_airy_ai` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 677 | `at::special_bessel_j0` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 678 | `at::special_bessel_j1` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 679 | `at::special_bessel_y0` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 680 | `at::special_bessel_y1` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 681 | `at::special_chebyshev_polynomial_t` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 682 | `at::special_chebyshev_polynomial_u` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 683 | `at::special_chebyshev_polynomial_v` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 684 | `at::special_chebyshev_polynomial_w` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 685 | `at::special_digamma` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 686 | `at::special_entr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 687 | `at::special_erf` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 688 | `at::special_erfc` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 689 | `at::special_erfcx` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 690 | `at::special_erfinv` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 691 | `at::special_exp2` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 692 | `at::special_expit` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 693 | `at::special_expm1` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 694 | `at::special_gammainc` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 695 | `at::special_gammaincc` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 696 | `at::special_gammaln` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 697 | `at::special_hermite_polynomial_h` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 698 | `at::special_hermite_polynomial_he` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 699 | `at::special_i0` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 700 | `at::special_i0e` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 701 | `at::special_i1` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 702 | `at::special_i1e` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 703 | `at::special_laguerre_polynomial_l` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 704 | `at::special_legendre_polynomial_p` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 705 | `at::special_log1p` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 706 | `at::special_log_ndtr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 707 | `at::special_log_softmax` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 708 | `at::special_logit` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 709 | `at::special_logsumexp` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 710 | `at::special_modified_bessel_i0` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 711 | `at::special_modified_bessel_i1` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 712 | `at::special_modified_bessel_k0` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 713 | `at::special_modified_bessel_k1` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 714 | `at::special_multigammaln` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 715 | `at::special_ndtr` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 716 | `at::special_ndtri` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 717 | `at::special_polygamma` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 718 | `at::special_psi` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 719 | `at::special_round` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 720 | `at::special_scaled_modified_bessel_k0` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 721 | `at::special_scaled_modified_bessel_k1` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 722 | `at::special_shifted_chebyshev_polynomial_t` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 723 | `at::special_shifted_chebyshev_polynomial_u` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 724 | `at::special_shifted_chebyshev_polynomial_v` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 725 | `at::special_shifted_chebyshev_polynomial_w` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 726 | `at::special_sinc` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 727 | `at::special_softmax` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 728 | `at::special_spherical_bessel_j0` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 729 | `at::special_xlog1py` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 730 | `at::special_xlogy` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 731 | `at::special_zeta` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 732 | `at::split_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 733 | `at::split_with_sizes_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 734 | `at::squeeze_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 735 | `at::sspaddmm` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 736 | `at::std_mean` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 737 | `at::stride` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 738 | `at::sub` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 739 | `at::sum_to_size` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 740 | `at::swapaxes` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 741 | `at::swapdims` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 742 | `at::sym_constrain_range` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 743 | `at::sym_constrain_range_for_size` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 744 | `at::sym_is_contiguous` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 745 | `at::sym_numel` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 746 | `at::sym_size` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 747 | `at::sym_storage_offset` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 748 | `at::sym_stride` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 749 | `at::t_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 750 | `at::take` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 751 | `at::take_along_dim` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 752 | `at::tensordot` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 753 | `at::thnn_conv2d` | - | 功能缺失 | 卷积操作变体 |
| 754 | `at::threshold` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 755 | `at::to_dense` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 756 | `at::to_padded_tensor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 757 | `at::transpose_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 758 | `at::trapezoid` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 759 | `at::trapz` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 760 | `at::triplet_margin_loss` | - | 功能缺失 | 损失函数，Paddle暂无直接对应实现 |
| 761 | `at::true_divide` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 762 | `at::type_as` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 763 | `at::unbind_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 764 | `at::unflatten_dense_tensors` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 765 | `at::unfold_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 766 | `at::unique_dim` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 767 | `at::unique_dim_consecutive` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 768 | `at::unsafe_chunk` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 769 | `at::unsqueeze_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 770 | `at::upsample_bicubic2d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 771 | `at::upsample_bilinear2d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 772 | `at::upsample_linear1d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 773 | `at::upsample_nearest1d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 774 | `at::upsample_nearest2d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 775 | `at::upsample_nearest3d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 776 | `at::upsample_trilinear3d` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 777 | `at::values` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 778 | `at::values_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 779 | `at::vander` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 780 | `at::var_mean` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 781 | `at::vdot` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 782 | `at::view_as_complex` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 783 | `at::view_as_complex_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 784 | `at::view_as_real` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 785 | `at::view_as_real_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 786 | `at::view_copy` | - | 功能缺失 | 稀疏张量copy操作变体 |
| 787 | `at::vstack` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 788 | `at::xlogy` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 789 | `at::xor` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 790 | `at::zero` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |
| 791 | `at::_aminmax` | - | 功能缺失 | PyTorch内部实现API |
| 792 | `at::_unique` | - | 功能缺失 | PyTorch内部实现API |
| 793 | `at::max_pool2d_with_indices` | - | 功能缺失 | 池化操作变体 |
| 794 | `at::max_pool3d_with_indices` | - | 功能缺失 | 池化操作变体 |
| 795 | `at::range` | - | 功能缺失 | PyTorch特有功能，Paddle暂无直接对应实现 |

## 统计

- **API 完全一致**: 66 个
- **仅 API 调用方式不一致**: 4 个
- **仅参数名不一致**: 77 个
- **paddle 参数更多**: 39 个
- **参数默认值不一致**: 2 个
- **torch 参数更多**: 19 个
- **输入参数用法不一致**: 0 个
- **输入参数类型不一致**: 37 个
- **返回参数类型不一致**: 26 个
- **组合替代实现**: 0 个
- **API 别名**: 13 个
- **语义差异**: 1 个
- **功能缺失**: 794 个
- **API 别名映射数**: 14 个
- **libtorch 主 ops 总数**: 1082 个
- **实际参与映射的 ops 数**: 1089 个
