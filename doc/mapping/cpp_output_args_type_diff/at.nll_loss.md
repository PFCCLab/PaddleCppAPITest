## [返回参数类型不一致]at::nll_loss

### PyTorch C++ API
```cpp
at::nll_loss(self, target, weight={}, reduction=at::Reduction::Mean, ignore_index=-100) -> Tensor
```

### Paddle C++ API
```cpp
paddle::experimental::nll_loss(input, label, weight, ignore_index=-100, reduction="mean") -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| Tensor | tuple<Tensor, Tensor> | 返回类型不一致。 |
