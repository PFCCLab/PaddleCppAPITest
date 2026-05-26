## [返回参数类型不一致]at::argsort

### PyTorch C++ API
```cpp
at::argsort(self, dim=-1, descending=false) -> Tensor
```

### Paddle C++ API
```cpp
paddle::experimental::argsort(x, axis=-1, descending=false, stable=false) -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| Tensor | tuple<Tensor, Tensor> | 返回类型不一致。 |
