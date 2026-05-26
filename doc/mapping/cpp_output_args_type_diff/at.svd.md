## [返回参数类型不一致]at::svd

### PyTorch C++ API
```cpp
at::svd(self, some=true, compute_uv=true) -> ::tuple<Tensor,Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::svd(x, full_matrices=false) -> tuple<Tensor, Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor,Tensor> | tuple<Tensor, Tensor, Tensor> | 返回类型不一致。 |
