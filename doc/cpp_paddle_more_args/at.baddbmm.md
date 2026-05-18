## [paddle 参数更多]at::baddbmm

### PyTorch C++ API
```cpp
at::baddbmm(self, batch1, batch2, beta=1, alpha=1)
```

### Paddle C++ API
```cpp
paddle::experimental::baddbmm(input, x, y, beta=1.0, alpha=1.0, out_dtype=DataType::UNDEFINED)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| batch1 | - | Paddle 无此参数，PyTorch 有 `batch1`。 |
| batch2 | - | Paddle 无此参数，PyTorch 有 `batch2`。 |
| beta | beta | 参数名一致。 |
| alpha | alpha | 参数名一致。 |
| - | input | PyTorch 无此参数，Paddle 有 `input`。 |
| - | y | PyTorch 无此参数，Paddle 有 `y`。 |
| - | out_dtype | PyTorch 无此参数，Paddle 有 `out_dtype`。 |
