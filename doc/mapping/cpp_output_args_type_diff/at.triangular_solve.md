## [返回参数类型不一致]at::triangular_solve

### PyTorch C++ API
```cpp
at::triangular_solve(self, A, upper=true, transpose=false, unitriangular=false) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::triangular_solve(x, y, upper=true, transpose=false, unitriangular=false) -> Tensor
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | Tensor | 返回类型不一致。 |
