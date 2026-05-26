## [仅参数名不一致]at::masked_scatter

### PyTorch C++ API
```cpp
at::masked_scatter(self, mask, source)
```

### Paddle C++ API
```cpp
paddle::experimental::masked_scatter(x, mask, value)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| mask | mask | 参数名一致。 |
| source | value | 仅参数名不一致，`source` 对应 `value`。 |
