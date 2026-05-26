## [仅参数名不一致]at::cholesky

### PyTorch C++ API
```cpp
at::cholesky(self, upper=false)
```

### Paddle C++ API
```cpp
paddle::experimental::cholesky(x, upper=false)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| upper | upper | 参数名一致。 |
