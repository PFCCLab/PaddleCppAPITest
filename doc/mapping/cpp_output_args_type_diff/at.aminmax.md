## [返回参数类型不一致]at::aminmax

### PyTorch C++ API
```cpp
at::aminmax(self, dim=::std::nullopt, keepdim=false) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::aminmax(x, axis={}, keepdim=false) -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | tuple<Tensor, Tensor> | 返回类型不一致。 |
