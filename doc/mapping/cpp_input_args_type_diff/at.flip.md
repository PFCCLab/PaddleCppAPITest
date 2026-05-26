## [输入参数类型不一致]at::flip

### PyTorch C++ API
```cpp
at::flip(self, dims)
```

### Paddle C++ API
```cpp
paddle::experimental::flip(x, axis)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dims | axis | 参数名与类型均不一致，PyTorch `dims` (`IntArray`) 对应 Paddle `axis` (`std::vector<int>`)。 |
