## [paddle 参数更多]at::logcumsumexp

### PyTorch C++ API
```cpp
at::logcumsumexp(self, dim)
```

### Paddle C++ API
```cpp
paddle::experimental::logcumsumexp(x, axis=-1, flatten=false, exclusive=false, reverse=false)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 默认值不同：PyTorch 无默认值，Paddle 默认 `axis=-1`。 |
| - | flatten | PyTorch 无此参数，Paddle 有 `flatten`。 |
| - | exclusive | PyTorch 无此参数，Paddle 有 `exclusive`。 |
| - | reverse | PyTorch 无此参数，Paddle 有 `reverse`。 |
