## [输入参数类型不一致]at::logit

### PyTorch C++ API
```cpp
at::logit(self, eps=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::logit(x, eps=1e-6)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| eps | eps | 参数类型不一致，PyTorch 为 `::optional<double>`，Paddle 为 `double`。 |
