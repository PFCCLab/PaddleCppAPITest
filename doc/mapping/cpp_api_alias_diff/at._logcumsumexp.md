## [API 别名]at::_logcumsumexp

### PyTorch C++ API
```cpp
at::_logcumsumexp(self, dim)
```

### Paddle C++ API
```cpp
paddle::experimental::logcumsumexp(x, axis=-1, flatten=false, exclusive=false, reverse=false)
```

两者功能一致，但 API 名称不同，PyTorch 为 `_logcumsumexp`，Paddle 为 `logcumsumexp`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 |
| - | flatten | PyTorch 无此参数，Paddle 有 `flatten`。 |
| - | exclusive | PyTorch 无此参数，Paddle 有 `exclusive`。 |
| - | reverse | PyTorch 无此参数，Paddle 有 `reverse`。 |
