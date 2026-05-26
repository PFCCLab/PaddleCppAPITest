## [API 别名]at::_stack

### PyTorch C++ API
```cpp
at::_stack(tensors, dim=0)
```

### Paddle C++ API
```cpp
paddle::experimental::stack(x, axis=0)
```

两者功能一致，但 API 名称不同，PyTorch 为 `_stack`，Paddle 为 `stack`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| tensors | - | Paddle 无此参数，PyTorch 有 `tensors`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 |
| - | x | PyTorch 无此参数，Paddle 有 `x`。 |
