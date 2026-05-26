## [输入参数类型不一致]at::clip

### PyTorch C++ API
```cpp
at::clip(self, min, max=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::clip(x, min, max)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| min | min | 参数类型不一致，PyTorch 为 `::optional<Scalar>`，Paddle 为 `Scalar`。 |
| max | max | 参数类型不一致，PyTorch 为 `::optional<Scalar>`，Paddle 为 `Scalar`。 |
