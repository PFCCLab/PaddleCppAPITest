## [仅参数名不一致]at::rsqrt

### PyTorch C++ API
```cpp
at::rsqrt(self)
```

### Paddle C++ API
```cpp
paddle::experimental::rsqrt(x)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
