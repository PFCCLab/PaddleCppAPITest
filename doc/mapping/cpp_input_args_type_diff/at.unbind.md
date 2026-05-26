## [输入参数类型不一致]at::unbind

### PyTorch C++ API
```cpp
at::unbind(self, dim=0)
```

### Paddle C++ API
```cpp
paddle::experimental::unbind(input, axis=0)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | input | 仅参数名不一致，`self` 对应 `input`。 |
| dim | axis | 参数名与类型均不一致，PyTorch `dim` (`int64_t`) 对应 Paddle `axis` (`int`)。 |
