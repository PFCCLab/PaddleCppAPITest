## [API 别名]at::range

### PyTorch C++ API
```cpp
at::range(start, end, options={})
```

### Paddle C++ API
```cpp
paddle::experimental::arange(start, end, step, dtype, place={})
```

两者功能一致，但 API 名称不同，PyTorch 为 `range`，Paddle 为 `arange`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| start | start | 参数名一致。 |
| end | end | 参数名一致。 |
| options | - | Paddle 无此参数，PyTorch 有 `options`。 |
| - | step | PyTorch 无此参数，Paddle 有 `step`。 |
| - | dtype | PyTorch 无此参数，Paddle 有 `dtype`。 |
| - | place | PyTorch 无此参数，Paddle 有 `place`。 |
