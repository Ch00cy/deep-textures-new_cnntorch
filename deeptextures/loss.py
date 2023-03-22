"""Loss functions.
"""
import torch
from torch.nn.functional import mse_loss    # 요소별 평균 제곱 오차를 측정.


def gramm(tnsr: torch.Tensor) -> torch.Tensor:  # torch.Tensor : 단일 데이터 유형의 요소를 포함하는 다차원 행렬.
    """Computes Gram matrix for the input batch tensor. # input batch -> gram matrix 계산

    Args:
        tnsr (torch.Tensor): input tensor of the Size.
        ([B(batch size), C(channel size), H(img height), W(img width]).

    Returns:
        G (torch.Tensor): output tensor of the Size([B, C, C]).
    """

    # input tensor size : [B, C, H, W]
    # (1) : C (channel)
    N = tnsr.size(1)  # Number of filters (size of feature maps) = C
    # (-2) : H (height) , (-1) : W (width)
    M = tnsr.size(-2) * tnsr.size(-1)  # size of the feature maps = H * W
    # (0) : B (batch)
    # Feature map: B, C, H, W ->  B, N, M
    # Size([B(batch), N(number of filter = feature map number), M(feature map size = w * H)]
    F = tnsr.view(tnsr.size(0), N, M)  # Size([B, N, M]) / reshape 와 같이 크기 변경(원소의 개수는 유지)
    # Gram matrix: B, N, M -> B, N, N
    # torch.bmm : Tensor(행렬)의 곱을 batch 단위로 처리 <->torch.mm: 단일 Tensor(행렬)로 계산
    # (b*n*m) * (b*m*p) = (b*n*p)
    # N(number of filter = feature map number) 배치행렬 * M(feature map size = w * H) 배치행렬
    G = F.bmm(F.transpose(1, 2))  # Size([B, N, N])
    return G


def gram_loss(input: torch.Tensor, target: torch.Tensor, weight: float = 1.0):  # init.py 에서 사용
    """Computes MSE Loss for 2 Gram matrices of the same type.  # 2개 Gram 에 대해 MSE(평균제곱오차) loss 계산

    Args:
        input (torch.Tensor):
        target (torch.Tensor):
        weight (float):

    Returns
        loss (torch.Tensor): computed loss value.
    """
    # tensor 모양 -> [B(batch size), C(channel size), H(img height), W(img width]

    # input , target 2가지의 torch.Tensor 들어옴
    # torch.Tensor.size() : (mini-batch size(데이터 한 묶음당 들어있는 데이터쌍(x-y) 개수), channel size, img size, img size)
    Bi, Bt = input.size(0), target.size(0)  # 각각 input , target 의 미니 배치 사이즈
    assert Bi == Bt # assert 조건(true/false), 메세지(생략): 가정 설정문 , true 아닐 시 error (둘이 같아야 ture)

    Ni, Nt = input.size(1), target.size(1)  # 각각 input, target 의 채널 사이즈
    assert Ni == Nt # assert 조건(true/false), 메세지(생략): 가정 설정문 , true 아닐 시 error (둘이 같아야 ture)

    Mi, Mt = input.size(-2) * input.size(-1), target.size(-2) * target.size(-1) # 각각 img size의 width * height
    assert Mi == Mt # assert 조건(true/false), 메세지(생략): 가정 설정문 , true 아닐 시 error (둘이 같아야 ture)

    B, N, M = Bi, Ni, Mi    # input 에 대한 Batch size , Number of feturemap , Matrix-feturemap size(H * W)

    # tensor 모양 -> [B(batch size), C(channel size), H(img height), W(img width]

    Gi, Gt = gramm(input), gramm(target)    # 위의 gramm 함수 이용 -> gram 구함

    # mse = ( 1/n ) ( sum(Y1 - Y2)^2 )
    # El = (1 / (4 * N ** 2 * M ** 2)) * mse_loss(Gi, Gt, reduction="sum")
    # total Loss L = layer Sum(weight * E)
    loss = weight * (1 / (4 * N ** 2 * M ** 2)) * mse_loss(Gi, Gt, reduction="sum") / B
    # Batch도 사용돼서 한번 할 때 batch 만큼 나눠서 loss 가져가나봄..
    return loss
