## [API 别名]at::_aminmax

### PyTorch C++ API
```cpp
at::_aminmax(self)
```

### Paddle C++ API
```cpp
paddle::experimental::aminmax(x, axis={}, keepdim=false)
```

两者功能一致，但 API 名称不同，PyTorch 为 `_aminmax`，Paddle 为 `aminmax`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| - | axis | PyTorch 无此参数，Paddle 有 `axis`。 |
| - | keepdim | PyTorch 无此参数，Paddle 有 `keepdim`。 |
