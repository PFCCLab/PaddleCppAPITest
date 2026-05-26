## [返回参数类型不一致]at::where

### PyTorch C++ API
```cpp
at::where(condition) -> std::vector<Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::where(condition, x, y) -> Tensor
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| std::vector<Tensor> | Tensor | 返回类型不一致。 |
