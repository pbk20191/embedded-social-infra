# Project Title
2024학년도 졸업작품
## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

Write about 1-2 paragraphs describing the purpose of your project.

이 프로젝트는 finn, PYNQ, arm32 가상환경으로 구성되어 있습니다.

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

이 환경은 WSL2에서 구성되었으며, 리눅스는 확인하지는 않았으나, 가능할 것으로 보입니다.


[PYNQ](PYNQ)를 사용하려면 `Petalinux`, `Vitis`가 호스트 경로상에서 `/tools/Xilinx`에 설치가 되어 있어야 합니다. 아니면 [도커설정](PYNQ/.devcontainer/devcontainer.json)에서 `mounts` 옵션을 적절하게 변경해야합니다. 

자세한 내용은 PYNQ sdbuild instruction을 참고해주시기 바랍니다.

[finn](finn)를 사용하려면 `Vitis`가 호스트에 설치되어 있어야 합니다. estimation level의 기능까지는 `vitis` 없이도 사용이 가능하지만, 그 이후부터는 `vitis`가 반드시 필요합니다. 

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
