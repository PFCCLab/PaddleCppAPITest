## [返回参数类型不一致]at::median

### PyTorch C++ API
```cpp
at::median(self) -> Tensor
```

### Paddle C++ API
```cpp
paddle::experimental::median(x, axis={}, keepdim=true, mode="avg") -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| Tensor | tuple<Tensor, Tensor> | 返回类型不一致。 |
