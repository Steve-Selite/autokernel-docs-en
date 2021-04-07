# AutoKernel: Quick start of development environment

In this section, we will introduce how to install/configure the development environment of AutoKernel, and introduce two major components of this project: Tengine, Halide. In order to reduce the problems encountered by developers in configuring the environment, we currently provide Docker to configure the basic environment required. More environment configuration methods will be provided in the future.  

- AutoKernel Development Environment   
  - [AutoKernel install wizard](#autokernel-install wizard)
  - [Halide](#halide)
  - [Tengine](#tengine)
-------------------

## AutoKernel install wizard     
AutoKernel provides a docker image and a development environment for AutoKernel.    

- If you has not installed docker, please see [here](https://docs.docker.com/engine/install/debian/).   

- If you are not familiar with Docker，please see here: [Tutorials for docker users](https://www.runoob.com/docker/docker-hello-world.html)

Next we think you have installed docker.     

1. Pull the image (may take a while, please wait patiently, depending on the network speed, it may take 10-20mins)    
    ```
    docker pull openailab/autokernel
    ```
2. Create a container and enter the development environment     
    ```
    docker run -ti openailab/autokernel /bin/bash 
    ```
    Enter into the docker container     
    ```
    root@39bfb5ea515d:/workspace#
    ```
    * Note that if you have already created a container, you only need to start the container and enter. Otherwise, your previous changes will not take effect in the newly created container.     

    To view the container created before, you can rename your container with the command `docker container rename`, here, our container is called `autokernel`       
    ```
    $ docker container ls -a
    CONTAINER ID        IMAGE                  COMMAND             CREATED             STATUS                       PORTS               NAMES
    ff8b59212784        openailab/autokernel   "/bin/bash"         21 hours ago        Exited (255) 2 minutes ago                       autokernel
    ```

    Start container    
    ```
    docker start autokernel
    ```
    Enter container   
    ```
    docker exec -ti autokernel /bin/bash
    ```
3. Halide, Tengine have been installed in docker    
    ```
    /workspace/Halide	# Halide
    /workspace/Tengine  # Tengine
    ```

4. Clone AutoKernel     
    ```
    git clone https://github.com/OAID/AutoKernel.git
    ```

By now, the environmental documents we need later have been prepared.     

## Halide
Halide is a DSL programming language, which separates the algorithm from the hardware backend. This project will use Halide's DSL and IR. Halide has been installed in docker, and the Python API has been configured.    

Halide related files are all in the `/workspace/Halide/` folder, and the Halide installation files are all in the `/workspace/Halide/halide-build` folder.     

```
cd /workspace/Halide/halide-build
```
* Halide related files are all in the`/workspace/Halide/halide-build/include`
    ```
    root@bd3faab0f079:/workspace/Halide/halide-build/include# ls

    Halide.h                     HalideRuntimeHexagonDma.h
    HalideBuffer.h               HalideRuntimeHexagonHost.h
    HalidePyTorchCudaHelpers.h   HalideRuntimeMetal.h
    HalidePyTorchHelpers.h       HalideRuntimeOpenCL.h
    HalideRuntime.h              HalideRuntimeOpenGL.h
    HalideRuntimeCuda.h          HalideRuntimeOpenGLCompute.h
    HalideRuntimeD3D12Compute.h  HalideRuntimeQurt.h
    ```
* The compiled Halide library is in `/workspace/Halide/halide-build/src` directory, where we can find `libHalide.so` 
    ```
    root@bd3faab0f079:/workspace/Halide/halide-build/src# ls 
    CMakeFiles           autoschedulers       libHalide.so.10
    CTestTestfile.cmake  cmake_install.cmake  libHalide.so.10.0.0
    Makefile             libHalide.so         runtime
    ```
* Run Halide
    ```
    cd /workspace/Halide/halide-build
    ./tutorial/lesson_01_basics 
    ```
    execution result    
    ```
    Success!
    ```
* Run Halide's Python interface       
    First check the system path of Python     
    ```
    python
    >>>import sys
    >>> sys.path
    ['', '/root', '/workspace/Halide/halide-build/python_bindings/src', '/usr/lib/python36.zip', '/usr/lib/python3.6', '/usr/lib/python3.6/lib-dynload', '/usr/local/lib/python3.6/dist-packages', '/usr/lib/python3/dist-packages']
    ```
    You can see that the Python system path already has Halide's compiled python package path`'/workspace/Halide/halide-build/python_bindings/src'`
    ```
    python
    >>> import halide
    ```
    `import halide` means success！



## Tengine
Tengine is a lightweight high-performance deep neural network inference engine. This project will be based on Tengine for operator development and optimization.   

Tengine has been installed in docker, and related files are in the `/workspace/Tengine/` directory   
```
cd /workspace/Tengine/build
```
* Tengine related documents are in`/workspace/Tengine/build/install/include`
    ```
    root@bd3faab0f079:/workspace/Tengine/build/install/include# ls

    tengine_c_api.h
    tengine_cpp_api.h
    ```
* The compiled Tengine library is uner`/workspace/Tengine/build/install/lib` directory, where we ca find `libtengine-lite.so` 
    ```
    root@bd3faab0f079:/workspace/Tengine/build/install/lib# ls 

    libtengine-lite.so
    ```
* Run Tengine

    This example ran the performance benchmark of each network model of Tengine on the target computer     
    ```
    cd /workspace/Tengine/benchmark
    ../build/benchmark/tm_benchmark
    ```
    execution result    
    ```
    start to run register cpu allocator
    loop_counts = 1
    num_threads = 1
    power       = 0
    tengine-lite library version: 1.0-dev
        squeezenet_v1.1  min =   32.74 ms   max =   32.74 ms   avg =   32.74 ms
            mobilenetv1  min =   31.33 ms   max =   31.33 ms   avg =   31.33 ms
            mobilenetv2  min =   35.55 ms   max =   35.55 ms   avg =   35.55 ms
            mobilenetv3  min =   37.65 ms   max =   37.65 ms   avg =   37.65 ms
            shufflenetv2  min =   10.93 ms   max =   10.93 ms   avg =   10.93 ms
                resnet18  min =   74.53 ms   max =   74.53 ms   avg =   74.53 ms
                resnet50  min =  175.55 ms   max =  175.55 ms   avg =  175.55 ms
            googlenet  min =  133.23 ms   max =  133.23 ms   avg =  133.23 ms
            inceptionv3  min =  298.22 ms   max =  298.22 ms   avg =  298.22 ms
                vgg16  min =  555.60 ms   max =  555.60 ms   avg =  555.60 ms
                    mssd  min =   69.41 ms   max =   69.41 ms   avg =   69.41 ms
            retinaface  min =   13.14 ms   max =   13.14 ms   avg =   13.14 ms
            yolov3_tiny  min =  132.67 ms   max =  132.67 ms   avg =  132.67 ms
        mobilefacenets  min =   14.95 ms   max =   14.95 ms   avg =   14.95 ms
    ALL TEST DONE
    ```
