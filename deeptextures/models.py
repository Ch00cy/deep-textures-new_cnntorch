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
from torchvision.models.vgg import cfgs, make_layers, model_urls
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


class VGG19(nn.ModuleDict): # torch.nn.ModuleDict : Model dictionary - (string: module) 매핑, 키-값 쌍의 iterable
    def __init__(self, avg_pool=True):  # 초기화 함수
        super().__init__()  # 부모 클래스 의 __init__ 불러옴 / 부모클래스 -> nn.ModuleDict 로 추정
        self.avg_pool = avg_pool    # 클래스의 변수

        self.layer_names = """
        conv1_1 conv1_2 pool1
        conv2_1 conv2_2 pool2
        conv3_1 conv3_2 conv3_3 conv3_4 pool3
        conv4_1 conv4_2 conv4_3 conv4_4 pool4
        conv5_1 conv5_2 conv5_3 conv5_4 pool5
        """.split()
        layers = filter(lambda m: not isinstance(m, nn.ReLU), make_layers(cfgs["E"]))
        layers = map(lambda m: nn.AvgPool2d(2, 2) if (isinstance(m, nn.MaxPool2d) and self.avg_pool) else m, layers)
        self.update(dict(zip(self.layer_names, layers)))

        for p in self.parameters():
            p.requires_grad_(False)

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


def vgg19(avg_pool: bool = True, pretrained: bool = True,):
    model = VGG19(avg_pool=avg_pool)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["vgg19"], progress=True)
        model.load_state_dict(state_dict)

    return model
