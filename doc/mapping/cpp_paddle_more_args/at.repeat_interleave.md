## [paddle 参数更多]at::repeat_interleave

### PyTorch C++ API
```cpp
at::repeat_interleave(repeats, output_size=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::repeat_interleave(x, repeats, axis, output_size=-1)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| repeats | repeats | 参数名一致。 |
| output_size | output_size | 参数名一致。 默认值不同：PyTorch 默认 `output_size=::std::nullopt`，Paddle 默认 `output_size=-1`。 |
| - | x | PyTorch 无此参数，Paddle 有 `x`。 |
| - | axis | PyTorch 无此参数，Paddle 有 `axis`。 |
