## [paddle 参数更多]at::tril_indices

### PyTorch C++ API
```cpp
at::tril_indices(row, col, offset=0, options=at::kLong)
```

### Paddle C++ API
```cpp
paddle::experimental::tril_indices(rows, cols, offset, dtype, place={})
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| row | rows | 仅参数名不一致，`row` 对应 `rows`。 |
| col | cols | 仅参数名不一致，`col` 对应 `cols`。 |
| offset | offset | 参数名一致。 默认值不同：PyTorch 默认 `offset=0`，Paddle 无默认值。 |
| options | - | Paddle 无此参数，PyTorch 有 `options`。 |
| - | dtype | PyTorch 无此参数，Paddle 有 `dtype`。 |
| - | place | PyTorch 无此参数，Paddle 有 `place`。 |
