## [输入参数类型不一致]at::nansum

### PyTorch C++ API
```cpp
at::nansum(self, dim=::std::nullopt, keepdim=false, dtype=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::nansum(x, axis={}, dtype=DataType::UNDEFINED, keepdim=false)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 参数名与类型均不一致，PyTorch `dim` (`at::OptionalIntArray`) 对应 Paddle `axis` (`IntArray`)。 |
| keepdim | dtype | 参数名与类型均不一致，PyTorch `keepdim` (`bool`) 对应 Paddle `dtype` (`DataType`)。 |
| dtype | keepdim | 参数名与类型均不一致，PyTorch `dtype` (`::optional<ScalarType>`) 对应 Paddle `keepdim` (`bool`)。 |
