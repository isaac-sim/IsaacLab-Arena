
FROM ubuntu:24.04

ENV force_color_prompt=yes
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/cloudci

RUN apt-get update
RUN apt-get install -qy jq
RUN apt-get install -qy curl
RUN apt-get install -qy wget
RUN apt-get install -qy unzip
RUN apt-get install -qy python3-pip
RUN apt-get install -qy openssh-client

# opentofu (instead of terraform)
# https://opentofu.org/docs/intro/install/deb/
RUN curl --proto '=https' --tlsv1.2 -fsSL https://get.opentofu.org/install-opentofu.sh -o install-opentofu.sh
RUN chmod +x install-opentofu.sh
RUN ./install-opentofu.sh --install-method deb

# aws cli
# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
WORKDIR /tmp
RUN case "$(dpkg --print-architecture)" in \
        amd64) curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" ;; \
        arm64) curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" ;; \
    esac
RUN unzip awscliv2.zip
RUN ./aws/install

# copy cloudci code into container
COPY . /cloudci

# customoize bash prompt
RUN echo "export PS1='\[\033[01;31m\][IsaacLab Arena Cloud CI]\[\033[00m\] \w $ '" >>  /root/.bashrc

WORKDIR /cloudci

ENTRYPOINT [ "/bin/sh", "-c" ]
