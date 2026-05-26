## [参数默认值不一致]at::roll

### PyTorch C++ API
```cpp
at::roll(self, shifts, dims={})
```

### Paddle C++ API
```cpp
paddle::experimental::roll(x, shifts={}, axis={})
```

两者功能一致，但参数默认值不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| shifts | shifts | 参数默认值不一致，PyTorch 为 无默认值，Paddle 为 `{}`。 |
| dims | axis | 仅参数名不一致，`dims` 对应 `axis`。 |
