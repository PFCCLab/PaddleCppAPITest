## [torch 参数更多]at::embedding

### PyTorch C++ API
```cpp
at::embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=false, sparse=false)
```

### Paddle C++ API
```cpp
paddle::experimental::embedding(x, weight, padding_idx=-1, sparse=false)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| weight | weight | 参数名一致，但位置顺序不同：PyTorch 第 1 个参数 weight 对应 Paddle 第 2 个参数 weight，调用时需按名传参或调换位置。 |
| indices | x | 仅参数名不一致，`indices` 对应 `x`。 |
| padding_idx | padding_idx | 参数名一致。 |
| scale_grad_by_freq | - | Paddle 无此参数，PyTorch 有 `scale_grad_by_freq`。 |
| sparse | sparse | 参数名一致。 |
