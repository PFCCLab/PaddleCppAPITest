## [语义差异]at::set

### PyTorch C++ API
```cpp
at::set(self)
```

### Paddle C++ API
```cpp
paddle::experimental::set(x, source, dims={}, stride={}, offset=0)
```

**注意：两者语义不同，不应视为等价 API。**

- PyTorch `set(self)` 将 tensor 的 storage **重置为空**（释放原有数据）。
- Paddle `set(x, source, dims, stride, offset)` 从 `source` tensor 复制 storage 并应用到 `x`，可自定义 `dims`、`stride` 和 `offset` 实现视图操作。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| - | source | Paddle 独有，指定数据来源 tensor。 |
| - | dims | Paddle 独有，指定目标维度。 |
| - | stride | Paddle 独有，指定目标步长。 |
| - | offset | Paddle 独有，指定 storage 偏移量。 |
