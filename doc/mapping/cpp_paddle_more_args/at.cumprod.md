## [paddle 参数更多]at::cumprod

### PyTorch C++ API
```cpp
at::cumprod(self, dim, dtype=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::cumprod(x, dim, exclusive=false, reverse=false)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | dim | 参数名一致。 |
| dtype | - | Paddle 无此参数，PyTorch 有 `dtype`。 |
| - | exclusive | PyTorch 无此参数，Paddle 有 `exclusive`。 |
| - | reverse | PyTorch 无此参数，Paddle 有 `reverse`。 |
