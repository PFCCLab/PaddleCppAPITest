## [语义差异]at::set

### PyTorch C++ API
```cpp
at::set(self)
```

### Paddle C++ API
```cpp
paddle::experimental::set(x, source, dims={}, stride={}, offset=0)
```

**注意：PyTorch `at::set` 与 Paddle `paddle::experimental::set` 语义不同，不应视为等价 API。**

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| - | source | PyTorch 无此参数，Paddle 有 `source`。 |
| - | dims | PyTorch 无此参数，Paddle 有 `dims`。 |
| - | stride | PyTorch 无此参数，Paddle 有 `stride`。 |
| - | offset | PyTorch 无此参数，Paddle 有 `offset`。 |
