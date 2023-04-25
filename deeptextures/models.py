"""Models.
"""
import torch
import torch.nn as nn
# torch.nn : 인스턴스화 시켜야함 -> attribute(클래스 내부에 포함되어있는 메소드,변수) 활용해 state 저장 가능
# torch.nn.fuctional : 인스턴스화 시킬 필요 없이 바로 입력값 받을 수 있음
# 인스턴스 : 클래스 -> 객체 -> 실체화 => 인스턴스
from torch.nn.functional import relu    # relu 요소 관련
#from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url  #에러대체
# Pytorch Hub-> 연구 재현성을 촉진하도록 설계된 사전 훈련된 모델 리포지토리
# load_state_dict_from_url : 주어진 URL에서 Torch 직렬화된 개체를 로드 -> Dict[str, Any] 반환
from torchvision.models.vgg import cfgs, make_layers #, model_urls
# cfg 들: GG 모델의 구조 정보 (각 vgg 모델마다 input, output 개수, 레이어 개수, pooling 언제 하는지 정보 들어있음)
'''
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
'''
# make_layers : conv2d relu poll 등 레이어 만들기
# model_urls : vgg 11 ~ 19 모델들에 대한 정보 url 들 담음

# 수정
# change from your model_urls to this
# from torchvision.models.vgg import VGG19_Weights
#
# dict_vgg19 = torch.utils.model_zoo.load_url(VGG19_Weights.IMAGENET1K_V1)

from torchvision.models import vgg19, VGG19_Weights

class VGG19(nn.ModuleDict): # torch.nn.ModuleDict : Model dictionary - (string: module) 매핑, 키-값 쌍의 iterable
    def __init__(self, avg_pool=True):  # 초기화 함수
        super().__init__()  # 부모 클래스 의 __init__ 불러옴 / 부모클래스 -> nn.ModuleDict 로 추정
        self.avg_pool = avg_pool    # 클래스의 변수

        self.layer_names = """  # 클래스의 변수
        conv1_1 conv1_2 pool1
        conv2_1 conv2_2 pool2
        conv3_1 conv3_2 conv3_3 conv3_4 pool3
        conv4_1 conv4_2 conv4_3 conv4_4 pool4
        conv5_1 conv5_2 conv5_3 conv5_4 pool5
        """.split() # 공백을 기준으로 쪼개줌 -> layer_names 에 ['conv1_1','conv1_2',...] 형태로 쪼개서 들어감

        # self. 으로 변수 선언 안되어 있으므로 해당 클래스 지역변수
        # filter() : 파이썬의 내장 함수, 여러 개의 데이터로 부터 (list나 tuple) 일부의 데이터만 추려낼 때 사용.
        # filter(조건 함수, 여러 데이터) -> 여러 데이터(리스트, 튜플) 에서 조건함수에 맞는 데이터만 뽑아낸다.
        # 조건함수는 lambda 로 대체
        # isinstance(확인하고자 하는 데이터 값, 확인하고자 하는 데이터 타입) -> true,false 반환
        # cfgs 'E'의 레이어 이름들이 m이 nn.ReLU 값이 아닌것을 layers라는 변수에 넣는다.
        layers = filter(lambda m: not isinstance(m, nn.ReLU), make_layers(cfgs["E"]))
        # map(함수, (리스트, 튜플)등의 데이터) : 데이터들을 함수에 넣어 연결한다.
        # 조건함수는 lambda 로 대체
        # isinstance(확인하고자 하는 데이터 값, 확인하고자 하는 데이터 타입) -> true,false 반환
        # 1) m = AvgPool2d(2, 2)
        # 2) if (isinstance(m, nn.MaxPool2d) and self.avg_pool) -> m = AvgPool2d(2, 2)와 nn.MaxPool2d 데이터 타입이 같고 + init 에서 avg_pool=true일 때-> 아무일도 없음
        # 3) else m -> 그렇지 않을 때 m = AvgPool2d(2, 2)와
        # 4) 레이어 이름들 중 렐루 아닌것을 집어넣은 layers와 map 함
        layers = map(lambda m: nn.AvgPool2d(2, 2) if (isinstance(m, nn.MaxPool2d) and self.avg_pool) else m, layers)
        # zip() : 길이가 같은 리스트 등의 요소를 묶어주는 함수
        # dict() : 딕셔너리 = dict(zip([키1, 키2], [값1, 값2])) 형태로 받음 => 키값 : layer_names , 값 : layers 형태 딕셔너리
        self.update(dict(zip(self.layer_names, layers)))

        for p in self.parameters():
            p.requires_grad_(False) # gradient 구하기 위해서는 tensor 의 속성을 requires_grad_=True로 설정
            # flase 일 경우 -> gradient 를 업데이트 하지 않고, dropout, batchnormalization 등이 적용되지 않습니다.

    def remap_state_dict(self, state_dict):
        original_names = "0 2 4 5 7 9 10 12 14 16 18 19 21 23 25 27 28 30 32 34 36".split()
        new_mapping = dict(zip(original_names, self.layer_names))
        # Need to copy
        new_state_dict = state_dict.copy()

        for k in state_dict.keys():
            if "classifier" in k:
                del new_state_dict[k]
                continue

            idx = k.split(".")[1]

            name = k.replace("features." + idx, new_mapping[idx])
            new_state_dict[name] = state_dict[k]
            del new_state_dict[k]

        return new_state_dict

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = self.remap_state_dict(state_dict)
        super().load_state_dict(state_dict, **kwargs)

    def forward(self, x, layers: list = None):
        layers = layers or self.keys()
        outputs = {"input": x}
        for name, layer in self.items():
            inp = outputs[[*outputs.keys()][-1]]
            out = relu(layer(inp)) if "pool" not in name else layer(inp)
            outputs.update({name: out})

            del outputs[[*outputs.keys()][-2]]

            if name in layers:
                yield outputs[name]


def vgg19(avg_pool: bool = True, pretrained: bool = True,): # init.py 에서 사용
    model = VGG19(avg_pool=avg_pool)    # VGG19 는 클래스 -> class VGG19(nn.ModuleDict)

    if pretrained:  # 매개변수로 pretrained 여부 받아왔음
        #state_dict = load_state_dict_from_url(model_urls["vgg19"], progress=True)
        #수정
        checkpoint = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'   #수정
        state_dict = load_state_dict_from_url(checkpoint, progress=True)    #수정

        model.load_state_dict(state_dict)
        # model.load_state_dict : 역직렬화된 state_dict를 사용, 모델의 매개변수들을 불러옴.
        # state_dict는 간단히 말해 각 체층을 매개변수 Tensor로 매핑한 Python 사전(dict) 객체.

    return model
