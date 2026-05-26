## [输入参数类型不一致]at::cross

### PyTorch C++ API
```cpp
at::cross(self, other, dim=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::cross(x, y, axis=9)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| other | y | 仅参数名不一致，`other` 对应 `y`。 |
| dim | axis | 参数名与类型均不一致，PyTorch `dim` (`::optional<int64_t>`) 对应 Paddle `axis` (`int`)。 |
