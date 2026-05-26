## [API 别名]at::_conj

### PyTorch C++ API
```cpp
at::_conj(self)
```

### Paddle C++ API
```cpp
paddle::experimental::conj(x)
```

两者功能一致，但 API 名称不同，PyTorch 为 `_conj`，Paddle 为 `conj`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
