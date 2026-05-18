## [仅参数名不一致]at::ones_like

### PyTorch C++ API
```cpp
at::ones_like(self, options={}, memory_format=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::ones_like(x, dtype=DataType::UNDEFINED, place={})
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| options | dtype | 仅参数名不一致，`options` 对应 `dtype`。 |
| memory_format | place | 仅参数名不一致，`memory_format` 对应 `place`。 |
