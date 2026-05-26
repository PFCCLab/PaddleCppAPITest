## [输入参数类型不一致]at::bilinear

### PyTorch C++ API
```cpp
at::bilinear(input1, input2, weight, bias={})
```

### Paddle C++ API
```cpp
paddle::experimental::bilinear(x, y, weight, bias)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input1 | x | 仅参数名不一致，`input1` 对应 `x`。 |
| input2 | y | 仅参数名不一致，`input2` 对应 `y`。 |
| weight | weight | 参数名与类型均一致。 |
| bias | bias | 参数类型不一致，PyTorch 为 `::optional<Tensor>`，Paddle 为 `optional<Tensor>`。 |
