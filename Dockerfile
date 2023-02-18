FROM nvidia/cuda:11.7.0-devel-ubuntu22.04 AS compile-image
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
WORKDIR /app

# Install required packages
RUN set -xe \
 && apt-get -y update \
 && apt-get install -y software-properties-common curl build-essential git \
 && apt-get -y update \
 && add-apt-repository universe \
 && apt-get -y update \
 && apt-get -y install python3 python3-pip \
 && apt-get clean

# Install Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

# Install apex
RUN pip install packaging==23.0 torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN set -xe  \
 && git clone https://github.com/NVIDIA/apex.git \
 && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

# Install python packages
COPY requirements.txt ./
RUN set -xe \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

CMD ["sleep", "inf"]
