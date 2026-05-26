## [输入参数类型不一致]at::gelu

### PyTorch C++ API
```cpp
at::gelu(self, approximate="none")
```

### Paddle C++ API
```cpp
paddle::experimental::gelu(x, approximate=false)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| approximate | approximate | 参数类型不一致，PyTorch 为 `c10::string_view`，Paddle 为 `bool`。 |
