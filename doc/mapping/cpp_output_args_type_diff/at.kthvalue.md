## [返回参数类型不一致]at::kthvalue

### PyTorch C++ API
```cpp
at::kthvalue(self, k, dim=-1, keepdim=false) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::kthvalue(x, k=1, axis=-1, keepdim=false) -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | tuple<Tensor, Tensor> | 返回类型不一致。 |
