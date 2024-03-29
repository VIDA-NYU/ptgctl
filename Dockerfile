FROM python:3.9-slim

ENV DIR=/src/lib/ptgctl

RUN apt-get update && apt-get install -y python3-opencv git && rm -rf /var/lib/apt/lists/*

#RUN pip install --no-cache-dir -U torch torchvision opencv-python  # for build convenience

ADD setup.py README.md LICENSE $DIR/
ADD ptgctl/__init__.py ptgctl/__version__.py $DIR/ptgctl/
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -U -e '/src/lib/ptgctl[all]'
ADD ptgctl/ $DIR/ptgctl/

WORKDIR /src/app
ENTRYPOINT ["python3"]
