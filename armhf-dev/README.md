# armhf target binary 컨테이너

## armhf 아키텍처 바이너리를 빌드하는 에뮬레이팅 컨테이너입니다.

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

PYNQ-Zynq는 armhf 아키텍처로 필요한 바이너리 배포반이 없는 경우가 더러 있습니다. 특히 pip-wheel이 없는 경우가 더러 있습니다. 하지만 플랫폼의 제한된 리소스 때문에, 플랫폼에서 build-from-source는 무겁고 느린 문제가 있습니다. 따라서 armhf 에뮬레이팅 환경을 구성해서 컨테이너내에서 바이너리를 빌드해서 사용하는 것이 더 좋습니다

## Getting Started <a name = "getting_started"></a>

devcontainer 환경으로 vscode에서 open_in_container를 하면 됩니다


### Installing

도커가 에뮬레이팅 할 수 있도록 qemu-user-static 바이너리를 커널 영역에 링크합니다. (한번만 하면 됩니다.)

```sh
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

vscode `open in devcontainer`로 접속해서 필요한 바이너리를 빌드하면 됩니다

## Usage <a name = "usage"></a>

예를 들어서 numpy 바이너리가 필요하다면...

```sh
git clone https://github.com/numpy/numpy.git && cd numpy && git submodule update --init
python setup.py bdist_wheel 
```