## [输入参数类型不一致]at::triu

### PyTorch C++ API
```cpp
at::triu(self, diagonal=0)
```

### Paddle C++ API
```cpp
paddle::experimental::triu(x, diagonal=0)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| diagonal | diagonal | 参数类型不一致，PyTorch 为 `int64_t`，Paddle 为 `int`。 |
