## [paddle 参数更多]at::var

### PyTorch C++ API
```cpp
at::var(self, unbiased)
```

### Paddle C++ API
```cpp
paddle::experimental::var(x, axis={}, keepdim=false, unbiased=true, correction=1)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：当前对比基于 PyTorch 最简重载 `var(self, unbiased)`。PyTorch 也存在带 `dim/keepdim/correction` 的完整重载。

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| unbiased | unbiased | 参数名一致。 默认值不同：PyTorch 无默认值，Paddle 默认 `unbiased=true`。 |
| - | axis | PyTorch 无此参数，Paddle 有 `axis`。 |
| - | keepdim | PyTorch 无此参数，Paddle 有 `keepdim`。 |
| - | correction | PyTorch 无此参数，Paddle 有 `correction`。 |
