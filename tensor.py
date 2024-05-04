import torch

# 텐서 생성
x = torch.tensor([1.0, 2.0, 3.0])  # 1차원 텐서
y = torch.tensor([[1, 2], [3, 4]])  # 2차원 텐서

# 텐서의 합
z = x + x
print("z = x + x:", z)

# 텐서의 요소별 곱
w = x * x
print("w = x * x:", w)

# 2차원 텐서와 스칼라의 곱
v = y * 2
print("v = y * 2:", v)

# 텐서의 평균
mean_x = x.mean()
print("mean of x:", mean_x)

# 텐서의 형태 변경
reshaped_y = y.view(4)
print("reshaped y:", reshaped_y)

# GPU에서 텐서 연산 (GPU가 사용 가능한 경우)
if torch.cuda.is_available():
    x_cuda = x.cuda()
    print("x on GPU:", x_cuda)