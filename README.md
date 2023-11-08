# Installation
1. First, the model must be downloaded:
Navigate to `/models` and run:
```sh
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin
```
2. Build the docker image using the provided Dockerfile:
```sh 
docker build -t gordon_ramsai . 
```
3. Run a docker container from the image:
```sh
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v $HOME/data:/data -p 50031:50031/tcp gordon_ramsai
```
> Step 3 must be run through a vscode terminal to use the proxy
