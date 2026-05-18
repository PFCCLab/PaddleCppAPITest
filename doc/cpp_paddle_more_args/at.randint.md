## [paddle 参数更多]at::randint

### PyTorch C++ API
```cpp
at::randint(high, size, options=at::kLong)
```

### Paddle C++ API
```cpp
paddle::experimental::randint(low, high, shape, dtype=DataType::INT64, place={})
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| high | high | 参数名一致。 |
| size | - | Paddle 无此参数，PyTorch 有 `size`。 |
| options | - | Paddle 无此参数，PyTorch 有 `options`。 |
| - | low | PyTorch 无此参数，Paddle 有 `low`。 |
| - | shape | PyTorch 无此参数，Paddle 有 `shape`。 |
| - | dtype | PyTorch 无此参数，Paddle 有 `dtype`。 |
| - | place | PyTorch 无此参数，Paddle 有 `place`。 |
