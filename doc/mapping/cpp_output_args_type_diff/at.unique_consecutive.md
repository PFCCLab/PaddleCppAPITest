## [返回参数类型不一致]at::unique_consecutive

### PyTorch C++ API
```cpp
at::unique_consecutive(self, return_inverse=false, return_counts=false, dim=::std::nullopt) -> ::tuple<Tensor,Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::unique_consecutive(x, return_inverse=false, return_counts=false, axis={}, dtype=DataType::FLOAT32) -> tuple<Tensor, Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor,Tensor> | tuple<Tensor, Tensor, Tensor> | 返回类型不一致。 |
