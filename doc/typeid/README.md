## typeid.h 类级兼容文档索引

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/typeid.h`
- `/home/may/pytorch/c10/util/typeid.h`

---

### 按类拆分文档

1. `type_identifier.md`
   - `caffe2::TypeIdentifier`
2. `type_meta_data.md`
   - `caffe2::detail::TypeMetaData`
3. `type_meta.md`
   - `caffe2::TypeMeta`
4. `uninitialized.md`
   - `caffe2::detail::_Uninitialized`
5. `guard_long_unique_dummy.md`
   - `caffe2::detail::_guard_long_unique_dummy<T>`
6. `is_paddle_fundamental.md`
   - `caffe2::detail::is_paddle_fundamental<T>`

---

### 说明

1. 每个文档均按 `TensorBase.h` 风格使用表格描述 API 兼容状态。
2. 每个函数条目均包含：
   - 实现方式是否完全一致
   - 语义是否一致
   - 具体实现方式说明
3. 综合总览仍保留在 `../type_meta.md`。
