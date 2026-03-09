##### 记录PaddleCPPAPITest仓库检测出来的接口不一致情况

# Allocator 类与 torch 存在差异

## 差异点列表

1.  **构造函数参数默认值**
2.  **拷贝语义**
3.  **`get_deleter()` 在默认构造后的返回值**
4.  **`clear()` 后 `get_deleter()` 的行为**
5.  **Device 类型和方法**
6.  **`allocation()` 方法**

---

涉及到的 PR：https://github.com/PFCCLab/PaddleCppAPITest/pull/42/changes#diff
