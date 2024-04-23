ARG VARIANT="jammy"

FROM --platform=linux/arm/v7 buildpack-deps:${VARIANT}
# FROM --platform=linux/arm64 mcr.microsoft.com/devcontainers/base:jammy
LABEL dev.containers.features="common"
LABEL org.opencontainers.image.ref.name=${VARIANT}
LABEL org.opencontainers.image.version=22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV TZ="Asia/Seoul"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo libatomic1 locales python3-pip python3-venv \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


RUN locale-gen ko_KR.UTF-8 en_US.UTF-8
RUN update-locale en_US.UTF-8
RUN dpkg-reconfigure --frontend noninteractive locales


RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-linux-armhf' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz
RUN mv code /usr/local/bin
CMD ["bash"]
