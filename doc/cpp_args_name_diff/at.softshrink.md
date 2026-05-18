## [仅参数名不一致]at::softshrink

### PyTorch C++ API
```cpp
at::softshrink(self, lambd=0.5)
```

### Paddle C++ API
```cpp
paddle::experimental::softshrink(x, threshold=0.5)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| lambd | threshold | 仅参数名不一致，`lambd` 对应 `threshold`。 |
