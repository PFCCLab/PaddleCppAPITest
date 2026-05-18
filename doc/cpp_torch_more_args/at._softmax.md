## [torch 参数更多]at::_softmax

### PyTorch C++ API
```cpp
at::_softmax(self, dim, half_to_float)
```

### Paddle C++ API
```cpp
paddle::experimental::softmax(x, axis)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 |
| half_to_float | - | Paddle 无此参数，PyTorch 有 `half_to_float`。 |
