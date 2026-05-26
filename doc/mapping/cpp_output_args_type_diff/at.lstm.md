## [返回参数类型不一致]at::lstm

### PyTorch C++ API
```cpp
at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first) -> ::tuple<Tensor,Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::lstm(input, h0, c0, weight, bias, use_peepholes=true, is_reverse=false, is_test=false, gate_activation="sigmoid", cell_activation="tanh", candidate_activation="tanh") -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor,Tensor> | tuple<Tensor, Tensor> | 返回类型不一致。 |
