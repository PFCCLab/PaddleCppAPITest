## [仅参数名不一致]at::hardswish

### PyTorch C++ API
```cpp
at::hardswish(self)
```

### Paddle C++ API
```cpp
paddle::experimental::hardswish(x)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
