## [返回参数类型不一致]at::histogram

### PyTorch C++ API
```cpp
at::histogram(self, bins, weight={}, density=false) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::histogram(input, weight, bins=100, min=0.0, max=0.0, density=false) -> Tensor
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | Tensor | 返回类型不一致。 |
