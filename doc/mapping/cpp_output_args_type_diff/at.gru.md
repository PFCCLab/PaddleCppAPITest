## [返回参数类型不一致]at::gru

### PyTorch C++ API
```cpp
at::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::gru(input, h0, weight, bias, activation="tanh", gate_activation="sigmoid", is_reverse=false, origin_mode=false, is_test=false) -> Tensor
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | Tensor | 返回类型不一致。 |
