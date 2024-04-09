
import torch.nn as nn
from collections import OrderedDict
import brevitas.nn as qnn
from brevitas.inject.defaults import Int8ActPerTensorFloat

def makeCnnModel(
        in_bit_width:int = 8,
        weight_bit_width:int = 2,
        act_bit_width:int = 2,
        channel_multiplier:int = 1
    ):
    # bit_width가 Quantization level parameter로 보임
    
    # https://github.com/Xilinx/brevitas/blob/master/src/brevitas/nn/__init__.py
    forward = OrderedDict()
    # forward["qid1"] = qnn.QuantIdentity(bit_width=in_bit_width)
    # 입력채널:1개, 출력채널:32개(필터의 개수), 커널(필터) 크기:3, stride=1(한번에 한픽셀씩 이동), padding=1(1만큼의 패딩이 입력 이미지의 주변에 한 픽셀씩 패딩)
    forward["conv1"] = qnn.QuantConv2d(1, 32 * channel_multiplier, kernel_size=3, stride=1, padding=1, weight_bit_width=weight_bit_width,
                                       input_bit_width=in_bit_width, input_quant=Int8ActPerTensorFloat)
    # 객체 self에 Relu레이어 추가
    forward["relu1"] = qnn.QuantReLU(bit_width=act_bit_width)
    # 커널(풀링 영역) 크기:2, stride=2(풀링 연산시 커널이 2픽셀씩 이동)
    forward["pool1"] = qnn.QuantMaxPool2d(kernel_size=2, stride=2)
    forward["conv2"] = qnn.QuantConv2d(32 * channel_multiplier, 64 * channel_multiplier, kernel_size=3, stride=1, padding=1, weight_bit_width=weight_bit_width)
    forward["relu2"] = qnn.QuantReLU(bit_width=act_bit_width)
    forward["pool2"] = qnn.QuantMaxPool2d(kernel_size=2, stride=2)
    # 컨볼루션 레이어의 출력을 완전연결 레이어에 전달하기 전에 데이터 평탄화
    forward["flatten"] = nn.Flatten()
    # 완전연결 레이어, 64개의 채널과 각 채널당 7*7크기의 특성 맵 존재, 이 크기의 평탄화된 입력이 완전 연결 레이어에 입력으로 들어감. 출력 특성의 크기는 128
    forward["fc1"] = qnn.QuantLinear(64 * channel_multiplier * 7 * 7, 128 * channel_multiplier, bias=True, weight_bit_width=weight_bit_width)
    forward["relu3"] = qnn.QuantReLU(bit_width=act_bit_width)
    forward["fc2"] = qnn.QuantLinear(128 * channel_multiplier, 10, bias=True, weight_bit_width=weight_bit_width)
    # 완전연결 레이어에서 활성화 함수를 softmax로 변경할시의 코드는 다음과 같다.
    # forward["fc1"] = qnn.QuantLinear(64 * 7 * 7, 128, bias=True)
    # forward["fc2"] = qnn.QuantReLU(128, 10, bias=True)
    # forward["softmax"] = nn.Softmax(dim=1)
    # 일반적으로 relu는 은닉층에서, softmax는 출력 레이어에서 사용한다.

    model = nn.Sequential(forward)
    return model