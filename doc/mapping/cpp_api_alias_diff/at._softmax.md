## [API 别名]at::_softmax

### PyTorch C++ API
```cpp
at::_softmax(self, dim, half_to_float)
```

### Paddle C++ API
```cpp
paddle::experimental::softmax(x, axis)
```

两者功能一致，但 API 名称不同，PyTorch 为 `_softmax`，Paddle 为 `softmax`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 |
| half_to_float | - | Paddle 无此参数，PyTorch 有 `half_to_float`。 |
