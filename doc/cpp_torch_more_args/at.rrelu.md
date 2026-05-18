## [torch 参数更多]at::rrelu

### PyTorch C++ API
```cpp
at::rrelu(self, lower=0.125, upper=0.3333333333333333, training=false, generator=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::rrelu(x, lower=1.0f/8, upper=1.0f/3, is_test=false)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| lower | lower | 参数名一致。 |
| upper | upper | 参数名一致。 |
| training | is_test | 语义取反，`training` 对应 `!is_test`（training=true 时 is_test=false）。 |
| generator | - | Paddle 无此参数，PyTorch 有 `generator`。 |
