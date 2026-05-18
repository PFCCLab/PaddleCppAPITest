## [torch 参数更多]at::log_softmax

### PyTorch C++ API
```cpp
at::log_softmax(self, dim, dtype=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::log_softmax(x, axis=-1)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 默认值不同：PyTorch 无默认值，Paddle 默认 `axis=-1`。 |
| dtype | - | Paddle 无此参数，PyTorch 有 `dtype`。 |
