## [torch 参数更多]at::multinomial

### PyTorch C++ API
```cpp
at::multinomial(self, num_samples, replacement=false, generator=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::multinomial(x, num_samples=1, replacement=false)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| num_samples | num_samples | 参数名一致。 默认值不同：PyTorch 无默认值，Paddle 默认 `num_samples=1`。 |
| replacement | replacement | 参数名一致。 |
| generator | - | Paddle 无此参数，PyTorch 有 `generator`。 |
