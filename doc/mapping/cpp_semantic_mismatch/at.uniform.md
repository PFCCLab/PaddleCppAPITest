## [语义差异]at::uniform

### PyTorch C++ API
```cpp
at::uniform(self, from=0, to=1, generator=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::uniform(shape, dtype, min, max, seed, place={})
```

**注意：PyTorch `at::uniform` 与 Paddle `paddle::experimental::uniform` 语义不同，不应视为等价 API。**

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | - | Paddle 无此参数，PyTorch 有 `self`。 |
| from | - | Paddle 无此参数，PyTorch 有 `from`。 |
| to | - | Paddle 无此参数，PyTorch 有 `to`。 |
| generator | - | Paddle 无此参数，PyTorch 有 `generator`。 |
| - | shape | PyTorch 无此参数，Paddle 有 `shape`。 |
| - | dtype | PyTorch 无此参数，Paddle 有 `dtype`。 |
| - | min | PyTorch 无此参数，Paddle 有 `min`。 |
| - | max | PyTorch 无此参数，Paddle 有 `max`。 |
| - | seed | PyTorch 无此参数，Paddle 有 `seed`。 |
| - | place | PyTorch 无此参数，Paddle 有 `place`。 |
