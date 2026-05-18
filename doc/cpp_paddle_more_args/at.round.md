## [paddle 参数更多]at::round

### PyTorch C++ API
```cpp
at::round(self)
```

### Paddle C++ API
```cpp
paddle::experimental::round(x, decimals=0)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| - | decimals | PyTorch 无此参数，Paddle 有 `decimals`。 |
