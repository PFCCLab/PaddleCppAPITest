## [仅参数名不一致]at::acos

### PyTorch C++ API
```cpp
at::acos(self)
```

### Paddle C++ API
```cpp
paddle::experimental::acos(x)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
