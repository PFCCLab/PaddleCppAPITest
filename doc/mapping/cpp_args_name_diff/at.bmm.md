## [仅参数名不一致]at::bmm

### PyTorch C++ API
```cpp
at::bmm(self, mat2)
```

### Paddle C++ API
```cpp
paddle::experimental::bmm(x, y)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| mat2 | y | 仅参数名不一致，`mat2` 对应 `y`。 |
