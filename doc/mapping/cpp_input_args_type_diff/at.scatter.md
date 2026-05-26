## [输入参数类型不一致]at::scatter

### PyTorch C++ API
```cpp
at::scatter(self, dim, index, src)
```

### Paddle C++ API
```cpp
paddle::experimental::scatter(x, index, updates, overwrite=true)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | index | 参数名与类型均不一致，PyTorch `dim` (`int64_t`) 对应 Paddle `index` (`Tensor`)。 |
| index | updates | 仅参数名不一致，`index` 对应 `updates`。 |
| src | overwrite | 参数名与类型均不一致，PyTorch `src` (`Tensor`) 对应 Paddle `overwrite` (`bool`)。 |
