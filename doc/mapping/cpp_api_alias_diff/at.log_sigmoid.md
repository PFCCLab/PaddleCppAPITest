## [API 别名]at::log_sigmoid

### PyTorch C++ API
```cpp
at::log_sigmoid(self)
```

### Paddle C++ API
```cpp
paddle::experimental::logsigmoid(x)
```

两者功能一致，但 API 名称不同，PyTorch 为 `log_sigmoid`，Paddle 为 `logsigmoid`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
