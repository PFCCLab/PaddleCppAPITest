## [仅参数名不一致]at::scatter

### PyTorch C++ API
```cpp
at::scatter(self, dim, index, src)
```

### Paddle C++ API
```cpp
paddle::experimental::scatter(x, index, updates, overwrite=true)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | index | 仅参数名不一致，`dim` 对应 `index`。 |
| index | updates | 仅参数名不一致，`index` 对应 `updates`。 |
| src | overwrite | 仅参数名不一致，`src` 对应 `overwrite`。 |
