## [输入参数类型不一致]at::broadcast_tensors

### PyTorch C++ API
```cpp
at::broadcast_tensors(tensors)
```

### Paddle C++ API
```cpp
paddle::experimental::broadcast_tensors(input)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| tensors | input | 参数名与类型均不一致，PyTorch `tensors` (`TensorList`) 对应 Paddle `input` (`std::vector<Tensor>`)。 |
