## [torch 参数更多]at::searchsorted

### PyTorch C++ API
```cpp
at::searchsorted(sorted_sequence, self, out_int32=false, right=false, side=::std::nullopt, sorter={})
```

### Paddle C++ API
```cpp
paddle::experimental::searchsorted(sorted_sequence, values, out_int32=false, right=false)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| sorted_sequence | sorted_sequence | 参数名一致。 |
| self | values | 仅参数名不一致，`self` 对应 `values`。 |
| out_int32 | out_int32 | 参数名一致。 |
| right | right | 参数名一致。 |
| side | - | Paddle 无此参数，PyTorch 有 `side`。 |
| sorter | - | Paddle 无此参数，PyTorch 有 `sorter`。 |
