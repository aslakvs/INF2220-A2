# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.07-py3

# Copy code into image
COPY . /data

WORKDIR /data

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install llama-cpp-python==0.1.65 --force-reinstall --upgrade --no-cache-dir
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs

# Install llama model
WORKDIR /data/models
RUN wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin

# Expose correct port
EXPOSE 50031

# Start server
WORKDIR /data/deploy-llm-project
CMD ["python3", "app.py"]

