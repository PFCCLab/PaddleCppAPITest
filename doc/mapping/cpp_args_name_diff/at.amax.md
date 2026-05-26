## [仅参数名不一致]at::amax

### PyTorch C++ API
```cpp
at::amax(self, dim={}, keepdim=false)
```

### Paddle C++ API
```cpp
paddle::experimental::amax(x, axis={}, keepdim=false)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 |
| keepdim | keepdim | 参数名一致。 |
