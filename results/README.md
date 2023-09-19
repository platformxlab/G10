# Specification of Program Outputs

This directory contains all the DNN training execution traces. It is also used as the output directory of the G10 program.

### DNN Training Execution Traces

We provided some profiles generated on our A100 GPU here. They are `cudnnXXX.txt`, `cudnnXXXInputPF.txt`, `cudnnXXXPF.txt`, and `cudnnXXXWorkSpace.txt` under every model's directory. The content of each trace file is either the profiled execution time or workspace memory consumption of every CUDA kernel. To correctly use the traces in our simulator, users need to specify the paths of these four files as the value of `orig_kernel_time_file`,  `pf_kernel_time_file`, `input_pf_kernel_time_file`, `workspace_size_file` in the corresponding simulation config file.

If the user uses `src/resources/genconfig.py` to generate config files, the traces in this folder will be automatically used.

### Program Outputs

After running one single experiment, the G10 program will generate several files under the corresponding `results/"$model_name"` directory. These files will have the following directory hierarchy:
```bash
results/"$model_name"
├── $batchSize-$baselineName-$additionalParameter
│   ├── statistics
│   │   ├── layers.config
│   │   ├── kernels.config
│   │   ├── tensors.config
│   │   ├── interval.config
│   │   ├── prefetch_guide.config
│   │   └── ...
│   ├── run.log
│   ├── sim_result.final
│   └── sim_result.KernelStall  # optional
│   └── ...
├── $batchSize-$baselineName-$additionalParameter_TensorPeriodLog.py
└── $batchSize-$baselineName-$additionalParameter_NNMemConsumptionLog.py
```


## layers.config
This file contains analyzed model operator/layer information. Users should see something like this:
```bash
______________________________________________________________________________
Layer ID:3; Name:Conv2d (512,32,149,149)
Next Layers:
Next Layer 0
Layer ID:4; Name:BatchNorm2d (512,32,147,147)
Previous Layers:
Previous Layer 0
Layer ID:2; Name:ReLU (512,32,149,149)
Input Tensor: tensor5 Is weight (global)?: 0, Size in byte: 1454964736, Range:2553532416--4008497152
Output Tensor: tensor20 Is weight (global)?: 0, Size in byte: 1416167424, Range:9828384768--11244552192
d_Input Tensor: tensor23 Is weight (global)?: 0, Size in byte: 1454964736, Range:11244589056--12699553792
d_Output Tensor: tensor25 Is weight (global)?: 0, Size in byte: 1416167424, Range:14115721216--15531888640
Weight Tensor: tensor21 Is weight (global)?: 1, Size in byte: 36864, Range:16384--53248
d_Weight Tensor: tensor22 Is weight (global)?: 0, Size in byte: 36864, Range:11244552192--11244589056
______________________________________________________________________________
```
## kernels.config
This file contains information for each scheduled GPU kernel in one training iteration. Users should see something like this:
```bash
____________________________________________________________________
Kernel ID: 4, Name: Conv2d_Forward
Parent Layer ID:3; Name:Conv2d
Execution Time: 16564224
(512,32,149,149)
Input Tensors:
tensor21 Is weight (global)?: 1, Size in byte: 36864, Range:16384--53248
tensor5 Is weight (global)?: 0, Size in byte: 1454964736, Range:2553532416--4008497152
Output Tensors:
tensor1891 Is weight (global)?: 0, Size in byte: 2871173120, Range:166677065728--169548238848
tensor20 Is weight (global)?: 0, Size in byte: 1416167424, Range:9828384768--11244552192
____________________________________________________________________
```

## tensors.config
This file contains information for all tensors. Users should see something like this:
```bash
tensor0 Is weight (global)?: 0, Size in byte: 549281792, Range:0--549281792
tensor1 Is weight (global)?: 0, Size in byte: 1454964736, Range:549281792--2004246528
tensor2 Is weight (global)?: 1, Size in byte: 4096, Range:0--4096
tensor3 Is weight (global)?: 0, Size in byte: 4096, Range:2004246528--2004250624
tensor4 Is weight (global)?: 0, Size in byte: 549281792, Range:2004250624--2553532416
tensor5 Is weight (global)?: 0, Size in byte: 1454964736, Range:2553532416--4008497152
tensor6 Is weight (global)?: 0, Size in byte: 1454964736, Range:4008497152--5463461888
```

## interval.config 
This file includes the results of the Tensor Vitality Analysis. The inactive periods ("Hiding intervals" in this file) and live periods are shown for every tensor. Users should see something like this:
```bash
_______________________________________________________________
tensor5 Is weight (global)?: 0, Size in byte: 1454964736, Range:2553532416--4008497152
Liveness: Birth: 2, Death: 924.
____________________________________________________________
tensor5 Is weight (global)?: 0, Size in byte: 1454964736, Range:2553532416--4008497152
Hidding Intervals:
interval 0: 5--------920
Estimated Time:5.19297e+07
interval 1: 921--------923
Estimated Time:10905.6
_______________________________________________________________
```

## prefetch_guide.config
This file includes the results of G10's Smart Tensor Migration Scheduling Algorithm. This output file is present only if the experiment config uses G10 (i.e., ``prefetch_lru''). Users should see something like this:
```bash
Issued Time: 144 Tensor: 873 G:o From: Not_Known, To: In_ssd
Issued Time: 144 Tensor: 864 G:x From: Not_Known, To: Not_present
Issued Time: 144 Tensor: 876 G:o From: Not_Known, To: In_gpu
Issued Time: 144 Tensor: 875 G:x From: Not_Known, To: In_gpu
Issued Time: 144 Tensor: 856 G:x From: Not_Known, To: Not_present
Issued Time: 144 Tensor: 872 G:o From: Not_Known, To: In_ssd
Issued Time: 145 Tensor: 886 G:x From: Not_Known, To: In_gpu
Issued Time: 145 Tensor: 884 G:x From: Not_Known, To: In_gpu
```

## run.log
This file contains the log for the performance simulation.

## sim_result.final
This file contains the final performance statistics for this experiment. Users should see something like this:
```bash
kernel_stat.iter1.in_transfer = 6354656
kernel_stat.iter1.cpu_pf = 658900
kernel_stat.iter1.ssd_pf = 4870
kernel_stat.iter1.unalloc_pf = 0
kernel_stat.iter1.exe_time = 73042083515
kernel_stat.iter1.ideal_exe_time = 71939325757
kernel_stat.iter1.pf_exe_time = 415807752634
kernel_stat.iter1.slowdown = 1.01533
kernel_stat.iter1.speedup = 5.69271
```
Note that except for FlashNeuron, we run the simulation for two training iterations and we choose the statistics for the second iteration (iter1) when evaluating performance. This is because other designs will consider scheduling evictions of global tensors before the end of one iteration and fetching them after the beginning of the next iteration. This will cause a difference in the beginning situation between the first iteration and the following iterations. 

## sim_result.KernelStall
This file is for analyzing the performance breakdown (fig 12-13).

## TensorPeriodLog.py
This file is for analyzing the inactive periods of tensors (fig 3-4).

## NNMemConsumptionLog.py
This file is for analyzing the memory consumption of DNN training (fig 2).

