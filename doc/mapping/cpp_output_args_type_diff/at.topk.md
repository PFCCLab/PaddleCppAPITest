## [返回参数类型不一致]at::topk

### PyTorch C++ API
```cpp
at::topk(self, k, dim=-1, largest=true, sorted=true) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::topk(x, k=1, axis=-1, largest=true, sorted=true) -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | tuple<Tensor, Tensor> | 返回类型不一致。 |
