## [输入参数类型不一致]at::index_select

### PyTorch C++ API
```cpp
at::index_select(self, dim, index)
```

### Paddle C++ API
```cpp
paddle::experimental::index_select(x, index, axis=0)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | index | 参数名与类型均不一致，PyTorch `dim` (`int64_t`) 对应 Paddle `index` (`Tensor`)。 |
| index | axis | 参数名与类型均不一致，PyTorch `index` (`Tensor`) 对应 Paddle `axis` (`int`)。 |
