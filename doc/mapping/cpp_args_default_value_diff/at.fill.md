## [参数默认值不一致]at::fill

### PyTorch C++ API
```cpp
at::fill(self, value)
```

### Paddle C++ API
```cpp
paddle::experimental::fill(x, value=0)
```

两者功能一致，但参数默认值不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| value | value | 参数默认值不一致，PyTorch 为 无默认值，Paddle 为 `0`。 |
