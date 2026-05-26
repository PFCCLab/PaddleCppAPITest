## [输入参数类型不一致]at::diag_embed

### PyTorch C++ API
```cpp
at::diag_embed(self, offset=0, dim1=-2, dim2=-1)
```

### Paddle C++ API
```cpp
paddle::experimental::diag_embed(input, offset=0, dim1=-2, dim2=-1)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | input | 仅参数名不一致，`self` 对应 `input`。 |
| offset | offset | 参数类型不一致，PyTorch 为 `int64_t`，Paddle 为 `int`。 |
| dim1 | dim1 | 参数类型不一致，PyTorch 为 `int64_t`，Paddle 为 `int`。 |
| dim2 | dim2 | 参数类型不一致，PyTorch 为 `int64_t`，Paddle 为 `int`。 |
