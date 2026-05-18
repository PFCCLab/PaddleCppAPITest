## [仅参数名不一致]at::tile

### PyTorch C++ API
```cpp
at::tile(self, dims)
```

### Paddle C++ API
```cpp
paddle::experimental::tile(x, repeat_times={})
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dims | repeat_times | 仅参数名不一致，`dims` 对应 `repeat_times`。 |
