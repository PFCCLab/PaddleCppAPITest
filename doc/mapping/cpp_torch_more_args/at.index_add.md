## [torch 参数更多]at::index_add

### PyTorch C++ API
```cpp
at::index_add(self, dim, index, source, alpha=1)
```

### Paddle C++ API
```cpp
paddle::experimental::index_add(x, index, add_value, axis=0)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 默认值不同：PyTorch 无默认值，Paddle 默认 `axis=0`。 |
| index | index | 参数名一致。 |
| source | add_value | 仅参数名不一致，`source` 对应 `add_value`。 |
| alpha | - | 影响计算语义，PyTorch 计算 self + alpha * other，Paddle 无此参数，等价表达需组合调用。 |
