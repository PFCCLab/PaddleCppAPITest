## [paddle 参数更多]at::lu_solve

### PyTorch C++ API
```cpp
at::lu_solve(self, LU_data, LU_pivots)
```

### Paddle C++ API
```cpp
paddle::experimental::lu_solve(b, lu, pivots, trans)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | - | Paddle 无此参数，PyTorch 有 `self`。 |
| LU_data | - | Paddle 无此参数，PyTorch 有 `LU_data`。 |
| LU_pivots | - | Paddle 无此参数，PyTorch 有 `LU_pivots`。 |
| - | b | PyTorch 无此参数，Paddle 有 `b`。 |
| - | lu | PyTorch 无此参数，Paddle 有 `lu`。 |
| - | pivots | PyTorch 无此参数，Paddle 有 `pivots`。 |
| - | trans | PyTorch 无此参数，Paddle 有 `trans`。 |
