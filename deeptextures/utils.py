"""Utilities.   # 도우미 , 추가기능
"""
import random   # 랜덤 함수 사용을 위한 import

import torch
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image
# resize : 입력 이미지를 주어진 크기로 조정 (이미지가 토치 텐서인 경우- […, H, W] 모양)
# to_tensor : PIL img , numpy.ndarray 를 텐서로 변환
# normalize : 평균 및 표준편차로 플로트 텐서 이미지를 정규화. 이 변환은 PIL 이미지를 지원하지 않음.
# to_pil_image : 텐서, ndarray를 PIL 이미지로 변환. 이 기능은 torchscript를 지원하지 않음.

from PIL import Image   # Python Imaging Library (pip install pillow)
# Image 모듈 : 기본적인 이미지 입출력 담당


MEAN = (0.485, 0.456, 0.406)    # 평균
STD = (0.229, 0.224, 0.225) # 표준편차


def set_seed(seed=None):    # <모델의 학습 파라미터와 테스트 값이 항상 달라지는 것을 최대한 방지 -> 재생산성을 위해>
    """Sets the random seed.    # seed 씨앗 : 랜덤값 줄 때 랜덤 기준값 , 상대가 seed값 알면 랜덤 깨짐
    """
    random.seed(seed)   # 무작위 난수 발생 , seed=None : 현재의 시간이 시드가 됨
    torch.manual_seed(seed) # 난수 생성을 위한 시드를 설정합니다. torch.Generator 객체를 반환합니다
    torch.backends.cudnn.deterministic = True   # Deterministic 한 알고리즘만 수행하게 하기. 불가능한 경우에는 RuntimeError
                                                # Deterministic(결정론적) : 선형회귀 로지스틱회귀 등이 여기 속함
    torch.backends.cudnn.benchmark = False  # cudnn은 convolution을 수행하는 과정에 벤치마킹을 통해서 지금 환경에 가장 적합한 알고리즘을 선정해 수행
                                            # 이 과정에서 다른 알고리즘이 선정되면 연산 후 값이 달라질 수 있는 것이다.
                                            # 이 설정을 켜 놓으면 성능 향상에 도움이 된다.


def set_device(device=None):    # gpu, cpu 설정
    """Sets the device.

    by default sets to gpu.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    # torch.cuda.is_available() : gpu 인식 하면 ture

def prep_img(image: str, size=None, mean=MEAN, std=STD):    # 이미지 미리 준비하기
    """Preprocess image.    이미지 전처리
    1) load as PIl  pillow사용 -> 이미지 load
    2) resize   사이즈 조정
    3) convert to tensor    텐서로 바꾸기
    4) normalize    정규화(정상화):모든 데이터가 동일한 정도의 스케일(중요도)로 반영되도록 해주는 것
    """
    im = Image.open(image)  # open
    size = size or im.size[::-1]    # 이미지 사이즈에서 처음부터 끝까지 -1칸 간격으로 (역순으로)
                                    # (width,height) -> (height,width)
    texture = resize(im, size)  # size 부분에 tuple 형태의 (w,h) - 튜플: () 로 둘러싸임
    texture_tensor = to_tensor(texture).unsqueeze(0)    # 이미지 -> 텐서형태로 -> unsqueeze(0): 0(첫번째 차원)에 1차원 더해줌
                                                        # squeeze : 차원이 1인 경우에는 해당 차원을 제거
                                                        # unsqueeze : 특정 위치에 1인 차원을 추가할 수 있습니다.

    # 텐서(tensor): 배열(array)이나 행렬(matrix)과 매우 유사한 특수한 자료구조 / NumPy의 ndarray 와 유사
    # 텐서 속성 : 텐서의 모양(shape), 자료형(datatype) , 어느 장치에 저장되는지(cpu,gpu)
    texture_tensor = normalize(texture_tensor, mean=mean, std=std)  # 정규화(정상화) 하는 함수 : 0~1 값 변경 및 스케일링
    # 인자:
    # tensor (Tensor) – <Float tensor image> of size 싱글데이터:(C채널, H, W) or 이미지:(B배치, C채널, H, W) to be normalized.
    # mean 평균 (sequence 연속성) – 각 채널에 대한 평균 시퀀스 (연속성)
    # std 표준편차 (sequence 연속성) – 각 채널에 대한 표준편차 시퀀스 (연속성)
    # inplace (bool,optional) – Bool to make this operation inplace. (새로운 텐서 리턴?)
    # output: 정규화 된 텐서 이미지
    return texture_tensor


def denormalize(tensor: torch.Tensor, mean=MEAN, std=STD, inplace: bool = False):
    """Based on torchvision.transforms.functional.normalize.
    """
    tensor = tensor.clone() if not inplace else tensor  # 기존 Tensor와 내용을 복사한 텐서 생성
    # 데이터(리스트)->텐서로 변환
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)   # mean 평균 데이터를 텐서로 => (?,1,1)크기로 변경
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1) # std 표준편차 데이터를 텐서로 => (?,1,1)크기로 변경
    tensor.mul_(std).add_(mean) # mul_:곱셈(기존 배열에 덮어씌워짐-> 리턴값 tensor에 저장) , add_ 자료는 없음..
    return tensor


def to_pil(tensor: torch.Tensor):   # 텐서 이미지 -> Pillow 형태 이미지
    """Converts tensor to PIL Image.

    Args:
        tensor (torch.Temsor): input tensor to be converted to PIL Image of torch.Size([C채널, H, W]).
    Returns:
        PIL Image: converted img.
    """
    img = tensor.clone().detach().cpu() # clone: 기존 Tensor 내용 복사 텐서
                                        # -> detach : gradient 전파가 안되는 텐서 생성(storage 공유 -> detach 텐서 변경시 원본 텐서 변경됨)
                                        # 파이토치- tensor에서 이루어진 모든 연산을 추적해서 기록(graph).
                                        # 이 연산 기록으로 부터 도함수가 계산되고 역전파가 이루어지게 된다.
                                        # detach()는 이 연산 기록으로 부터 분리한 tensor을 반환하는 method
                                        # cpu() :GPU 메모리에 올려져 있는 tensor를 cpu 메모리로 복사하는 method
    img = denormalize(img).clip(0, 1)   # 위에서 정의한 denormalize()로 정규화 푼다. -> clip() : 최소/최대 값 제한 -> 범위 (0,1)벗어나면 0,1 로 리턴함
    img = to_pil_image(img) # 텐서, ndarray를 PIL 이미지로 변환. 이 기능은 torchscript를 지원하지 않음.
    return img


def to_img(tensor):
    """To image tensor.
    """
    img = tensor.clone().detach().cpu() # 기존 텐서 내용 복사 -> 연산 기록 분리한 텐서 -> gpu 에서 cpu 로 복사
    img = denormalize(img).clip(0, 1)   # 위에서 정의한 denormalize()로 정규화 푼다. -> clip() : 최소/최대 값 제한 -> 범위 (0,1)벗어나면 0,1 로 리턴함
    img = img.permute(1, 2, 0)  # input크기가 순열된 원래 텐서의 뷰를 반환 / 입력.(원하는 차원 순서) -> 차원 순서 변경 가능 인듯?
    return img
