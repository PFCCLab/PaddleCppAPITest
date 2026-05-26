## [torch 参数更多]at::elu

### PyTorch C++ API
```cpp
at::elu(self, alpha=1, scale=1, input_scale=1)
```

### Paddle C++ API
```cpp
paddle::experimental::elu(x, alpha=1.0f)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| alpha | alpha | 参数名一致。 |
| scale | - | Paddle 无此参数，PyTorch 有 `scale`。 |
| input_scale | - | Paddle 无此参数，PyTorch 有 `input_scale`。 |
