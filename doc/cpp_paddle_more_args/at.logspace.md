## [paddle 参数更多]at::logspace

### PyTorch C++ API
```cpp
at::logspace(start, end, steps, base=10.0, options={})
```

### Paddle C++ API
```cpp
paddle::experimental::logspace(start, stop, num, base, dtype, place={})
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| start | start | 参数名一致。 |
| end | - | Paddle 无此参数，PyTorch 有 `end`。 |
| steps | - | Paddle 无此参数，PyTorch 有 `steps`。 |
| base | base | 参数名一致。 默认值不同：PyTorch 默认 `base=10.0`，Paddle 无默认值。 |
| options | - | Paddle 无此参数，PyTorch 有 `options`。 |
| - | stop | PyTorch 无此参数，Paddle 有 `stop`。 |
| - | num | PyTorch 无此参数，Paddle 有 `num`。 |
| - | dtype | PyTorch 无此参数，Paddle 有 `dtype`。 |
| - | place | PyTorch 无此参数，Paddle 有 `place`。 |
