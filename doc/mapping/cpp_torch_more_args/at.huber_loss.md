## [torch 参数更多]at::huber_loss

### PyTorch C++ API
```cpp
at::huber_loss(self, target, reduction=at::Reduction::Mean, delta=1.0)
```

### Paddle C++ API
```cpp
paddle::experimental::huber_loss(input, label, delta)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | input | 仅参数名不一致，`self` 对应 `input`。 |
| target | label | 仅参数名不一致，`target` 对应 `label`。 |
| reduction | - | Paddle 无此参数，PyTorch 有 `reduction`。 |
| delta | delta | 参数名一致。 默认值不同：PyTorch 默认 `delta=1.0`，Paddle 无默认值。 |
