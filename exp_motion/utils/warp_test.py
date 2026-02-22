import torch
import warp as wp

print("CUDA 可用:", torch.cuda.is_available())
print("Warp 初始化:", wp.init())

# 定义一个简单的 Warp 内核，执行基本的数学运算
@wp.kernel
def test_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = float(tid) * 2.0  # 显式转换 tid 为 float

# 创建一个数组作为内核的输入
n = 10
a = wp.zeros(n, dtype=wp.float32, device="cuda:0")

# 运行 Warp 内核
try:
    wp.launch(test_kernel, dim=n, inputs=[a], device="cuda:0")
    print("Warp 内核执行完毕，结果：", a.numpy())  # 将结果传回CPU并打印
except Exception as e:
    print("Warp CUDA 模块加载失败:", e)