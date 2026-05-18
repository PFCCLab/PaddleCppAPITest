## [torch 参数更多]at::gather

### PyTorch C++ API
```cpp
at::gather(self, dim, index, sparse_grad=false)
```

### Paddle C++ API
```cpp
paddle::experimental::gather(x, index, axis=0)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 默认值不同：PyTorch 无默认值，Paddle 默认 `axis=0`。 |
| index | index | 参数名一致。 |
| sparse_grad | - | Paddle 无此参数，PyTorch 有 `sparse_grad`。 |
