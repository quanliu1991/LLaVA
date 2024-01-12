FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
RUN apt-get update -y \
    && apt-get install -y python3-pip
USER root

RUN echo "Asia/Shanghai" > /etc/timezone
# Set the DEBIAN_FRONTEND environment variable to noninteractive
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y openssl openssh-server
RUN sed -i "s/#PubkeyAuthentication yes/PubkeyAuthentication yes/g" /etc/ssh/sshd_config
RUN sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/g" /etc/ssh/sshd_config
RUN sed -i "s/#Port 22/Port 22/g" /etc/ssh/sshd_config
#RUN sed -i '$a\service ssh start' /etc/bash.bashrc
RUN echo "root:root" | chpasswd
RUN echo -e '#!/bin/bash \nservice ssh start \n tail -f /dev/null'>/root/run.sh
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
WORKDIR /app
COPY . .
ENV LANG C.UTF-8
ENV PATH="/usr/local/python3.7.5/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH"
ENTRYPOINT ["bash", "/root/run.sh"]
CMD ["bash"]
WORKDIR app
ENV HOME="/app"
