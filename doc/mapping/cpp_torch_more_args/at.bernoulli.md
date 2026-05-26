## [torch 参数更多]at::bernoulli

### PyTorch C++ API
```cpp
at::bernoulli(self, generator=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::bernoulli(x)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| generator | - | Paddle 无此参数，PyTorch 有 `generator`。 |
