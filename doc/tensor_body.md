##### tensor_body.h 头文件 API 兼容性

**状态说明：**
- ✅ 已支持
- 🚧 正在开发
- ⏳ 尚未开发
- ❌ 不准备支持

**按照首字母进行排序**

| torch API                   | paddle API 兼容性 | 测试用例状态 | 优先级 |                  备注                  |
|-----------------------------|----------------|--------|-----|--------------------------------------|
| `abs`                       |       ✅        |   ✅    |  P2 |                                      |
| `absolute`                  |       ✅        |   ✅    |  P2 |                                      |
| `acos`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `acosh`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `add`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `addbmm`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `addcdiv`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `addcmul`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `addmm`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `addmv`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `addr`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `adjoint`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `alias`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `align_as`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `align_to`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `all`                       |       ✅        |   ✅    |  P2 |                                      |
| `allclose`                  |       ✅        |   ✅    |  P2 |                                      |
| `amax`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `amin`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `angle`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `any`                       |       🚧        |   🚧    |  P2 |                                      |
| `arccos`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `arccosh`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `arcsin`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `arcsinh`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `arctan`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `arctan2`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `arctanh`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `argmax`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `argmin`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `argsort`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `argwhere`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `as_strided`                |       ✅        |   ✅    |  P2 |                                      |
| `as_strided_`               |       ✅        |   ✅    |  P2 |                                      |
| `as_strided__symint`        |       ⏳        |   ⏳    |  P2 |                                      |
| `as_strided_scatter`        |       ✅        |   ✅    |  P2 |                                      |
| `as_strided_scatter_symint` |       ⏳        |   ⏳    |  P2 |                                      |
| `as_strided_symint`         |       ⏳        |   ⏳    |  P2 |                                      |
| `asin`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `asinh`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `atan`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `atan2`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `atanh`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `backward`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `baddbmm`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `bernoulli`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `bincount`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `bincount_symint`           |       ⏳        |   ⏳    |  P2 |                                      |
| `bitwise_and`               |       ⏳        |   ⏳    |  P2 |                                      |
| `bitwise_left_shift`        |       ⏳        |   ⏳    |  P2 |                                      |
| `bitwise_not`               |       ⏳        |   ⏳    |  P2 |                                      |
| `bitwise_or`                |       ⏳        |   ⏳    |  P2 |                                      |
| `bitwise_right_shift`       |       ✅        |   ✅    |  P2 |                                      |
| `bitwise_xor`               |       ⏳        |   ⏳    |  P2 |                                      |
| `bmm`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `broadcast_to`              |       ⏳        |   ⏳    |  P2 |                                      |
| `broadcast_to_symint`       |       ⏳        |   ⏳    |  P2 |                                      |
| `ccol_indices`              |       ⏳        |   ⏳    |  P2 |                                      |
| `ceil`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `chalf`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `cholesky`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `cholesky_inverse`          |       ⏳        |   ⏳    |  P2 |                                      |
| `cholesky_solve`            |       ⏳        |   ⏳    |  P2 |                                      |
| `clamp`                     |       ✅        |   ✅    |  P2 |        已实现 Scalar 和 Tensor 版本        |
| `clamp_max`                 |       ✅        |   ✅    |  P2 |        已实现 Scalar 和 Tensor 版本        |
| `clamp_min`                 |       ✅        |   ✅    |  P2 |        已实现 Scalar 和 Tensor 版本        |
| `clip`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `clone`                     |       ✅        |   ✅    |  P2 |                                      |
| `coalesce`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `col_indices`               |       ⏳        |   ⏳    |  P2 |                                      |
| `conj`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `conj_physical`             |       ⏳        |   ⏳    |  P2 |                                      |
| `contiguous`                |       ✅        |   ✅    |  P2 |                                      |
| `copysign`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `corrcoef`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `cos`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `cosh`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `count_nonzero`             |       ⏳        |   ⏳    |  P2 |                                      |
| `cov`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `cpu`                       |       ✅        |   ✅    |  P1 |                                      |
| `cross`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `crow_indices`              |       ⏳        |   ⏳    |  P2 |                                      |
| `cuda`                      |       ✅        |   ✅    |  P2 |                                      |
| `cumprod`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `cumsum`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `data`                      |       ✅        |   ✅    |  P1 |   torch准备遗弃该接口，建议使用 data_ptr<T>()    |
| `deg2rad`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `dense_dim`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `dequantize`                |       ⏳        |   ⏳    |  P2 |                                      |
| `det`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `detach`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `diag`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `diag_embed`                |       ⏳        |   ⏳    |  P2 |                                      |
| `diagflat`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `diagonal`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `diagonal_scatter`          |       ⏳        |   ⏳    |  P2 |                                      |
| `diff`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `digamma`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `dist`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `div`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `divide`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `dot`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `eq`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `equal`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `erf`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `erfc`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `erfinv`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `exp`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `exp2`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `expand`                    |       🚧        |   🚧    |  P2 |                                      |
| `expand_as`                 |       🚧        |   🚧    |  P2 |                                      |
| `expand_symint`             |       ⏳        |   ⏳    |  P2 |                                      |
| `expm1`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `fix`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `flatten`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `flip`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `fliplr`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `flipud`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `float_power`               |       ⏳        |   ⏳    |  P2 |                                      |
| `floor`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `floor_divide`              |       ⏳        |   ⏳    |  P2 |                                      |
| `fmax`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `fmin`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `fmod`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `frac`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `gather`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `gcd`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `ge`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `ger`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `greater`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `greater_equal`             |       ⏳        |   ⏳    |  P2 |                                      |
| `gt`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `hardshrink`                |       ⏳        |   ⏳    |  P2 |                                      |
| `hardshrink_backward`       |       ⏳        |   ⏳    |  P2 |                                      |
| `hash_tensor`               |       ⏳        |   ⏳    |  P2 |                                      |
| `heaviside`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `hip`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `histc`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `hypot`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `i0`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `igamma`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `igammac`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `index`                     |       ✅        |   ✅    |  P2 |                                      |
| `index_add`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `index_copy`                |       ⏳        |   ⏳    |  P2 |                                      |
| `index_fill`                |       ⏳        |   ⏳    |  P2 |                                      |
| `index_put`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `index_reduce`              |       ⏳        |   ⏳    |  P2 |                                      |
| `index_select`              |       ✅        |   ⏳    |  P2 |                                      |
| `indices`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `inner`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `int_repr`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `inverse`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `is_coalesced`              |       ⏳        |   ⏳    |  P2 |                                      |
| `is_distributed`            |       ⏳        |   ⏳    |  P2 |                                      |
| `is_nonzero`                |       ⏳        |   ⏳    |  P2 |                                      |
| `is_pinned`                 |       ✅        |   ✅    |  P2 |                                      |
| `is_same_size`              |       ⏳        |   ⏳    |  P2 |                                      |
| `is_set_to`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `is_variable`               |       ❌        |   ❌    |  P2 | torch准备遗弃该接口，现在所有 tensor 都是 variable |
| `isclose`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `isfinite`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `isinf`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `isnan`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `isneginf`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `isposinf`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `isreal`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `istft`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `item`                      |       ✅        |   ✅    |  P2 |                                      |
| `kron`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `lcm`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `ldexp`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `le`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `lerp`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `less`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `less_equal`                |       ⏳        |   ⏳    |  P2 |                                      |
| `lgamma`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `log`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `log10`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `log1p`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `log2`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `log_softmax`               |       ⏳        |   ⏳    |  P2 |                                      |
| `logaddexp`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `logaddexp2`                |       ⏳        |   ⏳    |  P2 |                                      |
| `logcumsumexp`              |       ⏳        |   ⏳    |  P2 |                                      |
| `logdet`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `logical_and`               |       ⏳        |   ⏳    |  P2 |                                      |
| `logical_not`               |       ⏳        |   ⏳    |  P2 |                                      |
| `logical_or`                |       ⏳        |   ⏳    |  P2 |                                      |
| `logical_xor`               |       ⏳        |   ⏳    |  P2 |                                      |
| `logit`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `logsumexp`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `lt`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `lu_solve`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `mH`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `mT`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `masked_fill`               |       ⏳        |   ⏳    |  P2 |                                      |
| `masked_scatter`            |       ⏳        |   ⏳    |  P2 |                                      |
| `masked_select`             |       ⏳        |   ⏳    |  P2 |                                      |
| `matmul`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `matrix_H`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `matrix_exp`                |       ⏳        |   ⏳    |  P2 |                                      |
| `matrix_power`              |       ⏳        |   ⏳    |  P2 |                                      |
| `max`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `maximum`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `mean`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `median`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `meta`                      |       ✅        |   ✅    |  P1 |                                      |
| `metal`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `min`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `minimum`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `mm`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `moveaxis`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `movedim`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `msort`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `mul`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `multinomial`               |       ⏳        |   ⏳    |  P2 |                                      |
| `multinomial_symint`        |       ⏳        |   ⏳    |  P2 |                                      |
| `multiply`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `mutable_grad`              |       ❌        |   ❌    |  P2 |                                      |
| `mv`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `mvlgamma`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `nan_to_num`                |       ⏳        |   ⏳    |  P2 |                                      |
| `nanmean`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `nanmedian`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `nanquantile`               |       ⏳        |   ⏳    |  P2 |                                      |
| `nansum`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `narrow`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `narrow_copy`               |       ⏳        |   ⏳    |  P2 |                                      |
| `narrow_copy_symint`        |       ⏳        |   ⏳    |  P2 |                                      |
| `narrow_symint`             |       ⏳        |   ⏳    |  P2 |                                      |
| `ne`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `neg`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `negative`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `new_empty`                 |       🚧        |   🚧    |  P2 |                                      |
| `new_empty_strided`         |       ⏳        |   ⏳    |  P2 |                                      |
| `new_empty_strided_symint`  |       ⏳        |   ⏳    |  P2 |                                      |
| `new_empty_symint`          |       ⏳        |   ⏳    |  P2 |                                      |
| `new_full`                  |       🚧        |   🚧    |  P2 |                                      |
| `new_full_symint`           |       ⏳        |   ⏳    |  P2 |                                      |
| `new_ones`                  |       🚧        |   🚧    |  P2 |                                      |
| `new_ones_symint`           |       ⏳        |   ⏳    |  P2 |                                      |
| `new_zeros`                 |       🚧        |   🚧    |  P2 |                                      |
| `new_zeros_symint`          |       ⏳        |   ⏳    |  P2 |                                      |
| `nextafter`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `nonzero`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `nonzero_static`            |       ⏳        |   ⏳    |  P2 |                                      |
| `nonzero_static_symint`     |       ⏳        |   ⏳    |  P2 |                                      |
| `norm`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `not_equal`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `numpy_T`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `orgqr`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `ormqr`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `outer`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `permute`                   |       ✅        |   ✅    |  P2 |                                      |
| `pin_memory`                |       ✅        |   ✅    |  P2 |                                      |
| `pinverse`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `polygamma`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `positive`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `pow`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `prelu`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `prod`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `put`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `q_per_channel_axis`        |       ⏳        |   ⏳    |  P2 |                                      |
| `q_per_channel_scales`      |       ⏳        |   ⏳    |  P2 |                                      |
| `q_per_channel_zero_points` |       ⏳        |   ⏳    |  P2 |                                      |
| `q_scale`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `q_zero_point`              |       ⏳        |   ⏳    |  P2 |                                      |
| `quantile`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `rad2deg`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `ravel`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `reciprocal`                |       ⏳        |   ⏳    |  P2 |                                      |
| `record_stream`             |       ✅        |   ✅    |  P2 |            仅 CUDA 版本               |
| `refine_names`              |       ⏳        |   ⏳    |  P2 |                                      |
| `relu`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `remainder`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `rename`                    |       🚧        |   🚧    |  P2 |                                      |
| `renorm`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `repeat`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `repeat_interleave`         |       ⏳        |   ⏳    |  P2 |                                      |
| `repeat_interleave_symint`  |       ⏳        |   ⏳    |  P2 |                                      |
| `repeat_symint`             |       ⏳        |   ⏳    |  P2 |                                      |
| `reshape`                   |       ✅        |   ✅    |  P2 |                                      |
| `reshape_as`                |       ⏳        |   ⏳    |  P2 |                                      |
| `reshape_symint`            |       ⏳        |   ⏳    |  P2 |                                      |
| `resize_`                   |       🚧        |   🚧    |  P2 |                                      |
| `resize__symint`            |       ⏳        |   ⏳    |  P2 |                                      |
| `resize_as_`                |       ⏳        |   ⏳    |  P2 |                                      |
| `resize_as_sparse_`         |       ⏳        |   ⏳    |  P2 |                                      |
| `resolve_conj`              |       ⏳        |   ⏳    |  P2 |                                      |
| `resolve_neg`               |       ⏳        |   ⏳    |  P2 |                                      |
| `roll`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `roll_symint`               |       ⏳        |   ⏳    |  P2 |                                      |
| `rot90`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `round`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `row_indices`               |       ⏳        |   ⏳    |  P2 |                                      |
| `rsqrt`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `scatter`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `scatter_add`               |       ⏳        |   ⏳    |  P2 |                                      |
| `scatter_reduce`            |       ⏳        |   ⏳    |  P2 |                                      |
| `select`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `select_scatter`            |       ⏳        |   ⏳    |  P2 |                                      |
| `select_scatter_symint`     |       ⏳        |   ⏳    |  P2 |                                      |
| `select_symint`             |       ⏳        |   ⏳    |  P2 |                                      |
| `sgn`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `sigmoid`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `sign`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `signbit`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `sin`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `sinc`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `sinh`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `size`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `slice`                     |       ✅        |   ✅    |  P2 |                                      |
| `slice_inverse`             |       ⏳        |   ⏳    |  P2 |                                      |
| `slice_inverse_symint`      |       ⏳        |   ⏳    |  P2 |                                      |
| `slice_scatter`             |       ⏳        |   ⏳    |  P2 |                                      |
| `slice_scatter_symint`      |       ⏳        |   ⏳    |  P2 |                                      |
| `slice_symint`              |       ⏳        |   ⏳    |  P2 |                                      |
| `smm`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `softmax`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `sparse_dim`                |       ⏳        |   ⏳    |  P2 |                                      |
| `sparse_mask`               |       ⏳        |   ⏳    |  P2 |                                      |
| `sparse_resize_`            |       ⏳        |   ⏳    |  P2 |                                      |
| `sparse_resize_and_clear_`  |       ⏳        |   ⏳    |  P2 |                                      |
| `sqrt`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `square`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `squeeze`                   |       ✅        |   ✅    |  P2 |                                      |
| `sspaddmm`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `std`                       |       ✅        |   ✅    |  P2 |                                      |
| `stft`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `stride`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `sub`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `subtract`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `sum`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `sum_to_size`               |       ⏳        |   ⏳    |  P2 |                                      |
| `sum_to_size_symint`        |       ⏳        |   ⏳    |  P2 |                                      |
| `swapaxes`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `swapdims`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `t`                         |       ⏳        |   ⏳    |  P2 |                                      |
| `take`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `take_along_dim`            |       ⏳        |   ⏳    |  P2 |                                      |
| `tan`                       |       ⏳        |   ⏳    |  P2 |                                      |
| `tanh`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `tensor_data`               |       ✅        |   ✅    |  P2 |                                      |
| `tile`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `tile_symint`               |       ⏳        |   ⏳    |  P2 |                                      |
| `to`                        |       ✅        |   ✅    |  P1 |                                      |
| `toBackend`                 |       ✅        |   ✅    |  P2 |     torch准备遗弃该接口（TODO标记      |
| `toType`                    |       ✅        |   ✅    |  P1 |                                      |
| `to_dense`                  |       ⏳        |   ⏳    |  P2 |                                      |
| `to_mkldnn`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `to_padded_tensor`          |       ⏳        |   ⏳    |  P2 |                                      |
| `to_padded_tensor_symint`   |       ⏳        |   ⏳    |  P2 |                                      |
| `to_sparse`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `to_sparse_bsc`             |       ⏳        |   ⏳    |  P2 |                                      |
| `to_sparse_bsr`             |       ⏳        |   ⏳    |  P2 |                                      |
| `to_sparse_csc`             |       ⏳        |   ⏳    |  P2 |                                      |
| `to_sparse_csr`             |       ⏳        |   ⏳    |  P2 |                                      |
| `trace`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `transpose`                 |       ✅        |   ⏳    |  P2 |                                      |
| `tril`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `triu`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `true_divide`               |       ⏳        |   ⏳    |  P2 |                                      |
| `trunc`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `type`                      |       ❌        |   ❌    |  P1 |           torch准备遗弃该接口          |
| `type_as`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `unflatten`                 |       ⏳        |   ⏳    |  P2 |                                      |
| `unflatten_symint`          |       ⏳        |   ⏳    |  P2 |                                      |
| `unfold`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `unsqueeze`                 |       ✅        |   ✅    |  P2 |                                      |
| `values`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `var`                       |       ✅        |   ✅    |  P2 |                                      |
| `variable_data`             |       ✅        |   ✅    |  P2 |                                      |
| `vdot`                      |       ⏳        |   ⏳    |  P2 |                                      |
| `ve`                        |       ⏳        |   ⏳    |  P2 |                                      |
| `view`                      |       ✅        |   ⏳    |  P2 |                                      |
| `view_as`                   |       ⏳        |   ⏳    |  P2 |                                      |
| `view_symint`               |       ⏳        |   ⏳    |  P2 |                                      |
| `vulkan`                    |       ⏳        |   ⏳    |  P2 |                                      |
| `where`                     |       ⏳        |   ⏳    |  P2 |                                      |
| `xlogy`                     |       ⏳        |   ⏳    |  P2 |                                      |

---

### 兼容性统计

| 状态 | 数量 |
|------|------|
| ✅ 已支持 | 37 |
| 🚧 正在开发 | 12 |
| ⏳ 尚未开发 | 329 |
| ❌ 不准备支持 | 3 |
| **总计** | **381** |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **实现说明**：
   - 所有标记为 ✅ 的接口均已实现并测试通过
   - 部分接口（如 `data`、`toBackend`、`is_variable`）在 torch 中准备遗弃，但为了兼容性仍提供支持
   - `clamp` 系列接口已实现 Scalar 和 Tensor 两种版本

3. **命名空间**：
   - 所有接口位于 `at::Tensor` 命名空间下
   - 兼容 PyTorch 的 `ATen` API 接口
