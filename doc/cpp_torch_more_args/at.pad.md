## [torch 参数更多]at::pad

### PyTorch C++ API
```cpp
at::pad(self, pad, mode="constant", value=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::pad(x, paddings, pad_value)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| pad | paddings | 仅参数名不一致，`pad` 对应 `paddings`。 |
| mode | - | Paddle 无此参数，PyTorch 有 `mode`。 |
| value | pad_value | 仅参数名不一致，`value` 对应 `pad_value`。 默认值不同：PyTorch 默认 `value=::std::nullopt`，Paddle 无默认值。 |
