##### 记录PaddleCPPAPITest仓库检测出来的接口不一致情况


1. Allocator类与torch存在差异
   差异点 1: 构造函数参数默认值
   差异点 2: 拷贝语义
   差异点 3: get_deleter() 在默认构造后的返回值
   差异点 4: clear() 后 get_deleter() 的行为
   差异点 5: Device 类型和方法
   差异点 6: allocation() 方法

    涉及到的PR：https://github.com/PFCCLab/PaddleCppAPITest/pull/42/changes#diff-c01b6db249bfc37591496432b46f774a297d197ab041a6fd1fe144ec363c9a85
