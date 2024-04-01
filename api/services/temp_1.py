import torch

# 创建一个形状为 (3, 3) 的 Tensor 对象
tensor = torch.tensor([[1, 2, 4], [2, 3, 4], [2, 2, 2]])

# 将第0维拆分为三个子 Tensor
tensor_list = torch.split(tensor, 1)

# 将每个子 Tensor 转换为列表
tensor_list = [t.squeeze().tolist() for t in tensor_list]

# 打印列表
print(tensor_list)