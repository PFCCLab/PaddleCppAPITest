## [仅参数名不一致]at::cholesky_solve

### PyTorch C++ API
```cpp
at::cholesky_solve(self, input2, upper=false)
```

### Paddle C++ API
```cpp
paddle::experimental::cholesky_solve(x, y, upper=false)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| input2 | y | 仅参数名不一致，`input2` 对应 `y`。 |
| upper | upper | 参数名一致。 |
