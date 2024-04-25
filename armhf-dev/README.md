# armhf target binary 컨테이너

## armhf 아키텍처 바이너리를 빌드하는 에뮬레이팅 컨테이너입니다.

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

PYNQ-Zynq는 armhf 아키텍처로 필요한 바이너리 배포반이 없는 경우가 더러 있습니다. 특히 pip-wheel이 없는 경우가 더러 있습니다. 하지만 플랫폼의 제한된 리소스 때문에, 플랫폼에서 build-from-source는 무겁고 느린 문제가 있습니다. 따라서 armhf 에뮬레이팅 환경을 구성해서 컨테이너내에서 바이너리를 빌드해서 사용하는 것이 더 좋습니다

## Getting Started <a name = "getting_started"></a>

devcontainer 환경으로 vscode에서 open_in_container를 하면 됩니다. 또는 devcontainer/cli가 있다면 이 폴더 경로 내에서 `devcontainer build --workspace-folder . && devcontainer up --mount-workspace-git-root=false && devcontainer cd /workspace/armhf-dev && bash`으로 컨테이너를 빌드 및 실행하고, 

### Prerequisites

What things you need to install the software and how to install them.

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo.

## Usage <a name = "usage"></a>

Add notes about how to use the system.
