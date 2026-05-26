## [paddle 参数更多]at::selu

### PyTorch C++ API
```cpp
at::selu(self)
```

### Paddle C++ API
```cpp
paddle::experimental::selu(x, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| - | scale | PyTorch 无此参数，Paddle 有 `scale`。 |
| - | alpha | PyTorch 无此参数，Paddle 有 `alpha`。 |
