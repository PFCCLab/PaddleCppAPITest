## [返回参数类型不一致]at::slogdet

### PyTorch C++ API
```cpp
at::slogdet(self) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::slogdet(x) -> Tensor
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | Tensor | 返回类型不一致。 |
