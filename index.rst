.. AutoKernel documentation master file, created by
   sphinx-quickstart on Wed Mar 11 12:03:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AutoKernel Docs!
===================================


.. toctree::
  :maxdepth: 1
  :caption: Introduction
  :name: sec-introduction

  introduction/architecture
  introduction/support_hardware
  introduction/support_op


.. toctree::
  :maxdepth: 1
  :caption: Quick Start
  :name: sec-quick-start

  quick_start/install
  quick_start/docker
  quick_start/tengine
  quick_start/halide
  quick_start/autokernel_plugin_tengine
  quick_start/gemm
  quick_start/cv_op
  quick_start/autosearch



.. toctree::
  :maxdepth: 1
  :caption: Multi Backend Demos
  :name: sec-demo_guides

  demo_guides/arm64_cpu
  demo_guides/x86_cpu
  demo_guides/opencl
  demo_guides/cuda_matmul/cuda

.. toctree::
  :maxdepth: 1
  :caption: Tutorials
  :name: sec-source-compile

  tutorials/01_AutoKernel_Quick_start_of_development_environment   
  tutorials/02_Tengine_Quick_start    
  tutorials/03_Halide_Quick_start   
  tutorials/04_AutoKernel_Plugin    
  tutorials/05_Halide_Schedule       
  tutorials/06_GEMM_How_to_optimize_schedule       


.. toctree::
  :maxdepth: 1
  :caption: Blog

  blog/ai_compiler overview   
  blog/autokernel_optimize_gemm_over_200_times_faster     

