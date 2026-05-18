## [paddle 参数更多]at::channel_shuffle

### PyTorch C++ API
```cpp
at::channel_shuffle(self, groups)
```

### Paddle C++ API
```cpp
paddle::experimental::channel_shuffle(x, groups, data_format="NCHW")
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| groups | groups | 参数名一致。 |
| - | data_format | PyTorch 无此参数，Paddle 有 `data_format`。 |
