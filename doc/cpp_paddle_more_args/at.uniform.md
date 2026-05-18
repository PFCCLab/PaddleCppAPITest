## [语义差异]at::uniform

### PyTorch C++ API
```cpp
at::uniform(self, from=0, to=1, generator=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::uniform(shape, dtype, min, max, seed, place={})
```

**注意：两者语义不同，不应视为等价 API。**

- PyTorch `uniform_` 是 **in-place 填充**操作，将调用 tensor 的值在 `[from, to)` 范围内均匀重初始化。
- Paddle `uniform` 是 **factory 函数**，根据给定 `shape` 创建一个新的 tensor 并填充均匀分布的值。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | - | PyTorch 的 inplace 操作对象，Paddle 无对应参数。 |
| from | min | 参数名不同，均表示均匀分布下界。 |
| to | max | 参数名不同，均表示均匀分布上界。 |
| generator | seed | PyTorch 使用 `generator`，Paddle 使用 `seed`。 |
| - | shape | Paddle 独有，指定输出 tensor 的形状。 |
| - | dtype | Paddle 独有，指定输出数据类型。 |
| - | place | Paddle 独有，指定设备位置。 |
