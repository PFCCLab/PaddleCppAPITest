## [paddle 参数更多]at::dropout

### PyTorch C++ API
```cpp
at::dropout(input, p, train)
```

### Paddle C++ API
```cpp
paddle::experimental::dropout(x, seed_tensor, p=0.5f, is_test=false, mode="downgrade_in_infer", seed=0, fix_seed=false)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input | x | 仅参数名不一致，`input` 对应 `x`。 |
| p | p | 参数名一致。 默认值不同：PyTorch 无默认值，Paddle 默认 `p=0.5f`。 |
| train | - | Paddle 无此参数，PyTorch 有 `train`。 |
| - | seed_tensor | PyTorch 无此参数，Paddle 有 `seed_tensor`。 |
| - | is_test | PyTorch 无此参数，Paddle 有 `is_test`。 |
| - | mode | PyTorch 无此参数，Paddle 有 `mode`。 |
| - | seed | PyTorch 无此参数，Paddle 有 `seed`。 |
| - | fix_seed | PyTorch 无此参数，Paddle 有 `fix_seed`。 |
