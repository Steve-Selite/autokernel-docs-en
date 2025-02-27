# AutoKernel Plugin    

In this section, we will introduce what AutoKernel Plugin is, the workflow of AutoKernel, and take Relu operator as an example to introduce how to add a new operator. In order to reduce the time for developers to add new operators, we currently provide template files and corresponding scripts for adding operators.     

##  What is AutoKernel Plugin    
Autokernel Plugin is a relatively independent plug-in, which only relies on the Tengine operator header file, and does not depend on the Tengine library. It realizes that the optimized operator code generated by AutoKernel Generator is integrated into the Tengine reasoning framework in the form of Plugin, and the one-click deployment of the automatic optimized operator is realized. The whole process does not need to recompile the Tengine library, only the dynamic library of Plugin needs to be compiled independently, the library of Autokernel Plugin is loaded at runtime, and the automatically generated operator implementation can be called.     

The following figure shows the changes before and after using AutoKernel, you only need to add a line of code at runtime to load the dynamic library of the autokernel plugin:    
```cpp
load_tengine_plugin();
```
![AutoKernel Plugin](data/plugin.png)


Work done by AutoKernel Plugin includes:
-The operator is encapsulated into Tengine's operator interface
-Register the operator into Tengine's operator library
-Compile and generate plugin dynamic library, load the dynamic library when calling


## Schematic

![add_op.png](data/add_op.png)

The workflow of AutoKernel is mainly divided into two steps:    
1. Generation: Write algorithm description and scheduling strategy, generate corresponding back-end optimization operator code.    

2. Deployment: Integrate the generated optimization operator code into Tengine in the form of plugin.   


The Autokernel project provides two implementations of the convolution operator   
-direct_conv: Direct convolution implementation    
-im2col_conv: convolution implementation of im2col+gemm     

Firstly, let's have a look at the directory structure:   
```bash
cd AutoKernel/autokernel_plugin/src
#if tree not installed, apt-get update & apt-get install tree
tree . 
```

We can see the directory structure of these two operators:    
```
.
|-- CMakeLists.txt
|-- direct_conv
|   |-- build.sh
|   |-- direct_conv.cpp
|   |-- direct_conv.h
|   `-- direct_conv_gen.cc
|-- im2col_conv
|   |-- build.sh
|   |-- im2col_conv.cpp
|   |-- im2col_conv.h
|   `-- im2col_conv_gen.cc
`-- plugin_init.cpp
```
* `xxx_gen.cpp`is a file used to generate code, which contains the operator calculation process and scheduling strategy described in Halide language.    
* `build.sh`is used to compile the generated file `xxx_gen.cpp` and formulate the output backend target.     
* `op.cpp`, `op.h`It is implemented by an operator encapsulated by Tengine's op interface, which calls an automatically generated operator function.      
* `plugin_init.cpp`It is used to register auto_op into Tengine's operator library.     


### 1.Generate   
`xxx_gen.cpp`It is a file used to generate code, which contains the operator calculation process and scheduling strategy described in Halide language.    
```cpp
// algorithm
Halide.Func(i,j)=...

// schedule
func.tile().reorder().parallel() ...
```
We provide the generated compilation script, taking the direct_conv operator as an example, the `build.sh` script is as follows：  
```bash
g++ direct_conv_gen.cc ../../common/GenGen.cpp \
	-I /workspace/Halide/halide-build/include/ \
	-L /workspace/Halide/halide-build/src \
	-lHalide -std=c++11 -fno-rtti \
	-o direct_conv_gen

./direct_conv_gen -g halide_direct_conv -e c_header,assembly -o . target=host
```
This script automatically links the Halide library and header files to generate the executable program `direct_conv_gen`. To perform the generation operation, you need to specify some of the generated parameters：    

- -g Name of function
- -e Options can be configured to generate multiple types of files. Here, assembly and header files are generated for tengine to call. The supported file types are as follows：
  
    [assembly, bitcode, cpp, h, html, o, static_library, stmt, cpp_stub, schedule, registration, featurization, pytorch_wrapper]

- -o Output path
- target Options are used to specify the type of backend, the options are as follows：   

    targets[] = {"arm-32-android", "arm-32-ios", "arm-32-linux", "arm-64-android", "arm-64-ios", arm-64-linux", "x86-32-linux", "x86-32-osx", "x86-32-windows", "x86-64-linux", "x86-64-osx", "x86-64-windows", "wasm-32-wasmrt"};

We provide one-click generation code for all operators   
```
cd AutoKernel/autokernel_plugin
chmod +x -R .
./scripts/generate.sh  #Automatically generate operator assembly file   
```
Check the src directory at this time, you can find that there are more automatically generated assembly files and header files in this directory    
```
.
|-- CMakeLists.txt
|-- direct_conv
|   |-- build.sh
|   |-- direct_conv.cpp
|   |-- direct_conv.h
|   |-- direct_conv_gen
|   |-- direct_conv_gen.cc
|   |-- halide_direct_conv.h
|   `-- halide_direct_conv.s
|-- im2col_conv
|   |-- build.sh
|   |-- halide_im2col_conv.h
|   |-- halide_im2col_conv.s
|   |-- im2col_conv.cpp
|   |-- im2col_conv.h
|   |-- im2col_conv_gen
|   `-- im2col_conv_gen.cc
`-- plugin_init.cpp
```
### 2.Deploy  

The deployment phase requires:   
1. Implement the operator encapsulated by Tengine's op interface and call the automatically generated operator function   
2. Register auto_op into Tengine's operator library   
3. One-click compilation of `libAutoKernel.so`   
4. Run the plugin dynamic library   


```
mkdir build
cd build
cmake ..
make -j4
```
Run test module
```
cd AutoKernel/autokernel_plugin
./build/tests/tm_classification -n squeezenet
```
Execution result：

```
AutoKernel plugin inited
function:autokernel_plugin_init executed

...

Repeat 1 times, avg time per run is 55.932 ms
max time is 55.932 ms, min time is 55.932 ms
--------------------------------------
0.2732 - "n02123045 tabby, tabby cat"
0.2676 - "n02123159 tiger cat"
0.1810 - "n02119789 kit fox, Vulpes macrotis"
0.0818 - "n02124075 Egyptian cat"
0.0724 - "n02085620 Chihuahua"
--------------------------------------
ALL TEST DONE
```
## Quickly add operators   
This tutorial will take the Relu operator as an example to demonstrate how to quickly develop the automatic optimization operator available for Tengine.    


### 1.Execute `register_op.sh` to automatically generate template files    
We provide a script file for quickly generating operators, and generate the source files and compilation scripts required for these two steps according to the template.     
```
cd AutoKernel/autokernel_plugin
chmod +x -R . 
./scripts/register_op.sh
```
Fill in according to the prompt：
```
op_name: relu
op_type: OP_RELU
```
The available file directories are as follows：
```
src/relu/relu.cpp
src/relu/relu.h
src/relu/relu_gen.cc
src/relu/build.sh
```
### 2.Generate: Edit the generated file`relu_gen.cc`
This file is used to generate operator assembly code. Use Halide language to describe the calculation process of the operator and the scheduling strategy schedule.   
In this example, schedule is empty by default.    

```
class halide_relu:public Halide::Generator<halide_relu>{
public:
    // args
    Input<Buffer<float>> input{"input", 4};
    Input<int> param{"param"};

    Output<Buffer<float>> output{"output", 4};

    void generate()
    {
        /* THE ALGORITHM */
        Var w("w"), h("h"), c("c"), n("n");
        Func halide_relu("halide_relu");
        halide_relu(w, h, c, n) = input(w, h, c, n);

        output(w, h, c, n) = select(param >= 0, max(param, halide_relu(w, h, c, n)), halide_relu(w, h, c, n));
    }

    void schedule()
    {
        /* THE SCHEDULE */
    }
};

```
### 3.Deployment: Edit `auto_relu.cpp`, one-click compilation to generate `AutoKernel.so`    

```
./scripts/generate.sh	# One-click to generate all the .s .h files needed by all operators       
mkdir build
cd build
cmake ..
make -j4
```

### 4.Test

The test case is for reference only [data/04_test_relu.cpp](data/04_test_relu.cpp)

```
#include "HalideBuffer.h"
#include <iostream>
#include "halide_relu.h"

int main(int argc, char **argv)
{
    int C = 1, W = 4, H = 4, N = 1;
    Halide::Runtime::Buffer<float> input_tensor(nullptr, W, H, C, N);
    Halide::Runtime::Buffer<float> output_tensor(nullptr, W, H, C, N);
    input_tensor.allocate();
    output_tensor.allocate();
    input_tensor.for_each_value([](float &x) {
        x = 2.0 * rand() / RAND_MAX - 1.0;
    });

    output_tensor.for_each_value([](float &x) {
        x = 2.0 * rand() / RAND_MAX - 1.0;
    });

    halide_relu(input_tensor, 0, output_tensor);

    printf("input:\n");
    for (int c = 0; c < input_tensor.dim(3).extent(); c++) {
        for (int z = 0; z < input_tensor.channels(); z++) {
            for (int y = 0; y < input_tensor.height(); y++) {
                for (int x = 0; x < input_tensor.width(); x++) {
                    std::cout<<input_tensor(x,y,z,0)<<" ";
                }
                std::cout<<"\n";
            }
            std::cout<<"\n";
        }
    }
    
    printf("output:\n");
    for (int c = 0; c < output_tensor.dim(3).extent(); c++) {
        for (int z = 0; z < output_tensor.channels(); z++) {
            for (int y = 0; y < output_tensor.height(); y++) {
                for (int x = 0; x < output_tensor.width(); x++) {
                    std::cout<<output_tensor(x,y,z,0)<<" ";
                }
                std::cout<<"\n";
            }
            std::cout<<"\n";
        }
    }

    return 0;
}
```
Put the test code `test_relu.cpp` in the AutoKernel/autokernel_plugin/build/ directory, and then compile the test case:    

```
g++ test_relu.cpp ../src/relu/halide_relu.s -I ../include/ -I ../src/relu/ -std=c++11 -lpthread -ldl -O3 -o relu_run
```
Execution
```
./relu_run
input:
0.680375 -0.211234 0.566198 0.59688 
0.823295 -0.604897 -0.329554 0.53645
-0.444451 0.10794 -0.0452059 0.25774
-0.270431 0.0268018 0.904459 0.83239

output:
0.680375 0 0.566198 0.59688 
0.823295 0 0 0.536459 
0 0.10794 0 0.257742 
0 0.0268018 0.904459 0.83239 
```


