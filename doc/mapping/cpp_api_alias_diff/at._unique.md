## [API 别名]at::_unique

### PyTorch C++ API
```cpp
at::_unique(self, sorted=true, return_inverse=false)
```

### Paddle C++ API
```cpp
paddle::experimental::unique(x, return_index, return_inverse, return_counts, axis, dtype=DataType::INT64)
```

两者功能一致，但 API 名称不同，PyTorch 为 `_unique`，Paddle 为 `unique`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| sorted | - | Paddle 无此参数，PyTorch 有 `sorted`。 |
| return_inverse | return_inverse | 参数名一致。 |
| - | return_index | PyTorch 无此参数，Paddle 有 `return_index`。 |
| - | return_counts | PyTorch 无此参数，Paddle 有 `return_counts`。 |
| - | axis | PyTorch 无此参数，Paddle 有 `axis`。 |
| - | dtype | PyTorch 无此参数，Paddle 有 `dtype`。 |
