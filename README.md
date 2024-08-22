# Persian RAG with multi user support

**Persian_RAG** is a Retrieval-Augmented Generation (RAG) application specifically designed for the Persian language. This project leverages advanced Language Models (LLMs) to enhance the accuracy and relevance of generated responses by integrating a retrieval mechanism to fetch contextual information.

## Table of Contents
- [Persian RAG with multi user support](#persian-rag-with-multi-user-support)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Architecture](#architecture)
  - [GPU Requirements](#gpu-requirements)
  - [Installation](#installation)

## Introduction

The goal of this project is to create a Persian specific RAG application that can be utilized for various NLP tasks such as question-answering, summarization, and more. By integrating retrieval mechanisms, the application is capable of providing more accurate and contextually relevant responses.

## Features

- **Persian Language Support**

- **Retrieval-Augmented Generation**: Combines retrieval-based and generation-based approaches for enhanced response accuracy.
- **Modular Design**: Easily extend or customize different components like the retrieval engine or generation model.
- **Dockerized Deployment**: Simple deployment using Docker.

## Architecture

The architecture of the Persian RAG system includes two main components:

1. **Retriever**: This component fetches relevant documents or passages from a large corpus based on the user's query.
2. **Generator**: This component generates the final response using the retrieved documents to ensure relevance and accuracy.
3. **Chainlit UI**: [Chainlit](https://github.com/Chainlit/chainlit) is on the best build production ready Conversational AI interfaces. I've used this to chat with my model.  

The RAG architecture ensures that the generation process is guided by factual and contextual data, improving the overall quality of the output.

## GPU Requirements
My workspace supprted me with a `nvidia A100 40GB`. but i assume the minimum requiremnts would be about `20GB` of VRAM.
## Installation

To get started with Persian_RAG, follow these steps:


1. Clone the repository:
    ```bash
    git clone https://github.com/shadmehr-salehi/Persian_Rag.git
    cd Persian_RAG
    ```

2. Build Dockerfiles for the engine and for the chainlit:
    ```bash
    docker build -t chainlitfront . 
    cd Ayaengine
    docker build -t ayaengine . 
    ```
3. Run the images 
    ```bash 
    docker run --gpus all --name ayaengine-f --port 5000:5000
    docker run --name chainfront --port 8000:8000
    ```
 

4.  make a docker volume named `llmxp_ayamodel` and put the model into `model` directory. `docker-compose` will be using it in the future. also create a volume named `chainlit_data` which will pass the data to the engine.<br>
    **NOTE** : you need to use `huggingface-cli` to download the model. <br> EG: `huggingface-cli download CohereForAI/aya-23-8B --local-dir .` when you are in the model directory
   
5. create a docker network `ayanet`.
6. `docker compose up` and your app is ready ! 
7. open `localhost:8000` and enjoy !