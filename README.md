# Tuned ruGPT3 on custom data

The following was used as initial data:

* Archive with digitized books by F.M. Dostoevsky
* Model ruGPT3small

The model was trained for five epochs, resulting in a model file of approximately 600 megabytes in size.

The specified file has been uploaded to the HuggingFace service and can be used locally for testing.

## Requirements

If you prefer the Docker way:

* Docker Engine
* Docker Compose
* Docker Nvidia Runtime
* CUDA 11.7

or if you prefer to install everything manually:

* Python 3.10
* CUDA 11.7
* NVCC

## How it was made

At the first step I've checked GitHub for projects in which was created custom
ruGPT3 model, which was trained on any text data

I've found [K7chyp/DostoevskyDoesntWriteIt](https://github.com/K7chyp/DostoevskyDoesntWriteIt) project, researched
sources and extracted commands, logic and prepared dataset with text.

Most important parts was copied to [train.sh](train.sh) and [prompt.sh](prompt.sh) scripts,
in general it was just a python scripts for executing pre-training and using pre-trained model, taken from original
ruGPT3 by [AI Forever](https://github.com/ai-forever/ru-gpts).

On next step I've tried to train own model with default parameters passed to `pretrain_transformers.py` and
found limitations of graphics card, 8Gb VRAM on my Nvidia RTX 3050 was not enough.

After several unsuccessful attempts, I managed to understand that changing the `block_size` parameter affects the amount
of memory used during model training. Therefore, I reduced it from 2048 to 512, after which the training was completed
without errors.

Next I've created Dockerfile and docker-compose.yml and project was done.

## How to install

Clone the repo, then switch working directory to sources root:

```shell
git clone git@github.com:EvilFreelancer/rugpt3-custom.git
cd rugpt3-custom
```

### The Doker way

Copy config:

```shell
cp docker-compose.dist.yml docker-compose.yml
```

Build and start:

```shell
docker-compose build
docker-compose up -d
```

Enter into container:

```shell
docker-compose exec app bash
```

### Manually

```shell
# Install packages
apt-get install -y software-properties-common curl build-essential git

# Install RUST
export PATH="~/.cargo/bin:${PATH}"
curl https://sh.rustup.rs -sSf | bash -s -- -y

# Install packages required for Apex
pip install packaging==23.0 torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Download and build Apex
export CUDA_HOME=/usr/local/cuda
git clone https://github.com/NVIDIA/apex.git
cd ./apex && git checkout 8b7a1ff183741dd8f9b87e7bafd04cfde99cea28 && cd ..
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

# Install ru-gpts
git clone https://github.com/EvilFreelancer/ru-gpts.git ru_gpts

# Install other dependencies
pip install -r requirements.txt

# For ruGPT3XL need to use requirements-xl.txt file
pip install -r requirements-xl.txt
```

## How to train (optional)

First you need to create train and validation data from [output.csv](./data/output.csv), for this need to execute:

```shell
python3 prepare.py
```

Then execute following script:

```shell
./train.sh
```

And wait for a some time.

Training on my Nvidia RTX 3050 took about 35 minutes, GPU temp 64&deg;С

## How to use

If you want to use your own model then exec following script:

```shell
./prompt.sh
```

But if you want to use my pretrained model uploaded to HuggingFace:

```shell
./prompt.hf.sh
```

After the model is loaded, you will see a command line prompt, just write a phrase and wait the result.

## Few examples

```
Москва, 19 июня /<18>69.  <…> У меня, например, есть один приятель, очень умный человек, но которого я непонимаю. Он
говорит мне:  –Знаете, Лев Николаич, я давно уже вас презирал, но вы, как человек умный, меня никогда не могли обидеть…
```

```
Однажды вечером, за обедом, я вдруг увидал, что у меня как будто все лицо изменяется: глаза смыкались, губы двигались;
нос тоже становился тоньше и суше, глаза сверкали и сверкали,– точно я что‑то предчувствовал и предугадывал. Я тотчас
же подошел к нему, поздоровался с ним, но он не ответил мне и только молча указал мне на стул, где я сидел. Я сел и
тотчас же опять начал его разглядывать. Он тотчас же потупил глаза и с минуту сидел неподвижно.
```

```
Меж тем он стал меня допрашивать.  –Ну, что же?– сказал я ему,– что же?  –А вот-с, что же-с!– отвечал он,– что же-с,
что ж?  –А вот что, Марья Александровна, что ж?– сказал я, немного покраснев от гнева,– что ж, что же? что же?  –Ах,
боже мой! Да ведь это все пустяки-с.
```

## Links

* https://huggingface.co/evilfreelancer/dostoevsky_doesnt_write_it
* https://github.com/K7chyp/DostoevskyDoesntWriteIt/
* https://github.com/ai-forever/ru-gpts
* https://github.com/GraphGrailAi/ruGPT3-ZhirV/
