## [torch 参数更多]at::binomial

### PyTorch C++ API
```cpp
at::binomial(count, prob, generator=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::binomial(count, prob)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| count | count | 参数名一致。 |
| prob | prob | 参数名一致。 |
| generator | - | Paddle 无此参数，PyTorch 有 `generator`。 |
