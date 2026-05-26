## [返回参数类型不一致]at::rms_norm

### PyTorch C++ API
```cpp
at::rms_norm(input, normalized_shape, weight={}, eps=::std::nullopt) -> Tensor
```

### Paddle C++ API
```cpp
paddle::experimental::rms_norm(x, scale, normalized_shape={}, epsilon=1.19209289550781250e-7) -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| Tensor | tuple<Tensor, Tensor> | 返回类型不一致。 |
