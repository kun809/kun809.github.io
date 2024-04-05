---
layout: post
title: "How to Create a Dockerized C++ Build Environment"
categories: misc
---
# Why do we need a dockerized environment
Many developers who work with C++ language, use GCC and CMake/Ninjia as their compiling tools. The tolls work very fine on Linux machines. But once switch to a new machine, they need to install and configure the tools manually. Despite the possibility of setting them up with a shell script in one step, discrepancies between Linux platforms may still lead to errors, consuming valuable time for resolution.

Fortunately, Docker can provide a consistent compilation environment among the Linux platform. And it is so easy to deploy and start to use it for our C++ project.

# How-to
## Before You Start
Before diving in, ensure you have a basic understanding of [Docker](https://www.docker.com) and access to a Linux machine with Docker installed.

## Start with a Dokerfile
Begin by creating a Dockerfile, building an image from it, and then launching a container.
A Dockerfile is a text document outlining the build steps for your environment. It specifies the Linux version, compiler, required packages, and any necessary system configurations, such as environment variables. Let's consider an example based on the Ubuntu 22.04 compilation environment:
```Dockerfile
# Declare the base image
FROM ubuntu:22.04

# Install the compiler and build tools
RUN apt update && apt install -y \
    gcc \
    g++ \
    cmake \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
```
* The FROM directive can specify any Linux distribution or custom tag as required.
* Including apt clean and rm -rf /var/lib/apt/lists/* reduces the image size.

Now, execute the docker build command to build the image:
```bash
docker build -t build-env-image -f <Dockerfile path> .
```
This will generate a Docker image:
```console
$ docker images
REPOSITORY        TAG       IMAGE ID       CREATED              SIZE
build-env-image   latest    4d016659e542   About a minute ago   367MB
```
You can start a container and verify the GCC version:
```console
$ docker run build-env-image gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
# Where Is My Source Code?
Typically, you'll need to mount your source folder when running the Docker container, enabling access and compilation of source code from within the container. 

To mount your source folder:
```bash
docker run -id --name build-env --mount type=bind,source=<your source folder>,target=<mount path>  build-env-image 
```
However, you can also fetch your source code using git or any other method within the container since it is like a "Linux virtual machine". 

I prefer to mount a host folder for ease of access with VS Code and to ensure that my code files are properly stored, although the Docker container functions more like a "temporary" environment.

The following diagram illustrates my compilation environment:

![My build environment](/assets/build_env_github_page.png)


