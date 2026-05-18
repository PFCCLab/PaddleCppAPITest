## [torch 参数更多]at::subtract

### PyTorch C++ API
```cpp
at::subtract(self, other, alpha=1)
```

### Paddle C++ API
```cpp
paddle::experimental::subtract(x, y)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| other | y | 仅参数名不一致，`other` 对应 `y`。 |
| alpha | - | 影响计算语义，PyTorch 计算 self - alpha * other，Paddle 无此参数，等价表达需组合调用。 |
