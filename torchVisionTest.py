import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 데이터셋을 로드하고 전처리하는 부분
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 사전 훈련된 ResNet 모델을 불러오는 부분
resnet = torchvision.models.resnet18(pretrained=True)
resnet.eval()

# 이미지를 인식하고 결과를 출력하는 부분
def predict_image(image):
    outputs = resnet(image)
    _, predicted = torch.max(outputs, 1)
    return predicted

# 이미지를 시각화하는 함수
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    # 테스트 이미지를 반복하여 인식하는 부분
    for images, labels in testloader:
        outputs = resnet(images)
        # softmax를 적용하여 확률로 변환
        probs = torch.nn.functional.softmax(outputs, dim=1)
        # 가장 높은 확률을 가진 클래스 선택
        _, predicted_class = torch.max(probs, 1)
        
        # 예측된 클래스 인덱스를 출력하여 확인
        print("Predicted class names:", predicted_class.names)

        # predicted_class를 텐서에서 리스트로 변환
        predicted_class = predicted_class.tolist()

          # 각 이미지에 대한 예측
        for i in range(len(images)):
            image = images[i].unsqueeze(0)  # 이미지 배치 차원 추가
            predicted_class = predict_image(image)
            
            # 입력 이미지와 모델이 예측한 클래스 출력
            print("Input Image:")
            print("Actual Class:", classes[labels[i]])
            print("왔>>", predicted_class.item())
        
        
        # for i in range(len(predicted_class)):
        #     print('Predicted:', classes[predicted_class[i]])
        #     # print('Actual:', classes[labels[i]])
        imshow(torchvision.utils.make_grid(images))