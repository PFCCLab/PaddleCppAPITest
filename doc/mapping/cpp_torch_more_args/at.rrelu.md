## [torch 参数更多]at::rrelu

### PyTorch C++ API
```cpp
at::rrelu(self, lower=0.125, upper=0.3333333333333333, training=false, generator=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::rrelu(x, lower=0.125f, upper=0.3333333333333333f, is_test=false)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| lower | lower | 参数名一致。 |
| upper | upper | 参数名一致。 |
| training | is_test | 【需对值取反】`training` 与 `is_test` 语义互为反义，不能直接搬运布尔值（training=true 时应设 is_test=false）。 |
| generator | - | Paddle 无此参数，PyTorch 有 `generator`。 |
