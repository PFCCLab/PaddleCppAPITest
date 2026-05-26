## [输入参数类型不一致]at::one_hot

### PyTorch C++ API
```cpp
at::one_hot(self, num_classes=-1)
```

### Paddle C++ API
```cpp
paddle::experimental::one_hot(x, num_classes)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| num_classes | num_classes | 参数类型不一致，PyTorch 为 `int64_t`，Paddle 为 `Scalar`。 |
