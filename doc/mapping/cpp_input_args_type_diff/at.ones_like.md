## [输入参数类型不一致]at::ones_like

### PyTorch C++ API
```cpp
at::ones_like(self, options={}, memory_format=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::ones_like(x, dtype=DataType::UNDEFINED, place={})
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| options | dtype | 参数名与类型均不一致，PyTorch `options` (`TensorOptions`) 对应 Paddle `dtype` (`DataType`)。 |
| memory_format | place | 参数名与类型均不一致，PyTorch `memory_format` (`::optional<at::MemoryFormat>`) 对应 Paddle `place` (`Place`)。 |
