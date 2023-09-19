from enum import Enum

#         modelname       : (netname,             config foldername,  output foldername,[batch sizes]) 
models = {# "Inceptionv3"   : ("Inceptionv3",       "Inceptionv3",      "Inceptionv3",    [512, 768, 1024, 1152, 1280, 1408, 1536, 1664, 1792]), 
		  "Inceptionv3"   : ("Inceptionv3",       "Inceptionv3",      "Inceptionv3",    [512, 768, 1024, 1152, 1280, 1536, 1792]), 
          
		  # "ResNet152"     : ("ResNet152",         "ResNet152",        "ResNet152",      [256, 512, 768, 1024, 1280, 1536, 1792, 2048]),
		  "ResNet152"     : ("ResNet152",         "ResNet152",        "ResNet152",      [256, 512, 768, 1024, 1280, 1536]),
          
		  # "ResNeXt101_32" : ("ResNeXt101_32x8d",  "ResNeXt101_32x8d", "ResNeXt101_32",  [256, 512, 768, 1024]),
          
		  # "SENet154"      : ("SENet154",          "SENet154",         "SENet154",       [256, 512, 768, 1024, 1280, 1536]),
		  "SENet154"      : ("SENet154",          "SENet154",         "SENet154",       [256, 512, 768, 1024]),
          
		  # "VIT"           : ("VIT",               "VIT",               "VIT",           [256, 512, 768, 1024, 1280, 1536, 1792, 2048]),
          "VIT"           : ("VIT",               "VIT",               "VIT",           [256, 512, 768, 1024, 1280, 1536]),
          
		  # "BERT"          : ("BERT_Base",         "BERT",             "BERT_Base",      [128, 256, 384, 512, 640, 768, 1024, 1280, 1536]),
          "BERT"          : ("BERT_Base",         "BERT",             "BERT_Base",      [128, 256, 384, 512, 640, 768, 1024]),
          
          }
policies = ["lru", "prefetch_lru", "deepUM", "FlashNeuron", "G10GDSSSD", "G10GDSFULL"] 
cpu_policies = ["lru", "prefetch_lru", "deepUM"]
ssd_policies = ["lru", "prefetch_lru", "deepUM", "FlashNeuron"]
hostmem_sizes = ["0", "16", "32", "64", "256"]
pcie_bws = ["32", "64", "128"]
ssd_bws = ["6.4", "12.8", "19.2", "25.6", "32"]
kernel_time_variations = ["0.05", "0.10", "0.15", "0.20", "0.25"]
kernel_speedup_variations = ["1.5", "2.0", "2.5", "3.0", "3.5", "4.0"]
kernel_speedup_policies = ["lru", "prefetch_lru", "deepUM", "FlashNeuron"]

# ===================
import os
current_folder = os.path.dirname(os.path.realpath(__file__)) # should be resources folder
config_basefolder = os.path.abspath(os.path.join(current_folder, os.path.pardir, "configs")) # should be configs folder
folder_basename = os.path.basename(current_folder)
if folder_basename != "resources":
  print(f"\033[0;31mInvalid current folder, <{current_folder}> is not <configs>\033[0m")
  exit(1)

config_output_folders = set()
config_names = set()
num_run_configs, num_profile_configs = 0, 0
# ===================

DEFAULT_HOSTMEM_SIZE = "128"
DEFAULT_SSD_BW = "3.2"
DEFAULT_PCIE_BW = "15.754"
DEFAULT_KERNEL_TIME_VARIATION = "0"
DEFAULT_KERNEL_SPEEDUP_VARIATION = "1"

def get_run_config_specification(model: str, netname: str, config_foldername: str, output_foldername: str, policy: str, batch_size: str, hostmem_size: str=DEFAULT_HOSTMEM_SIZE, pcie_bw : str=DEFAULT_PCIE_BW, ssd_bw : str=DEFAULT_SSD_BW, kernel_time_variation: str=DEFAULT_KERNEL_TIME_VARIATION, kernel_speedup_variation: str=DEFAULT_KERNEL_SPEEDUP_VARIATION):
  characteristics_str = ""
  if hostmem_size != DEFAULT_HOSTMEM_SIZE:
    characteristics_str += f"-cpu{hostmem_size.replace('.', '_')}"
  if ssd_bw != DEFAULT_SSD_BW:
    characteristics_str += f"-ssd{ssd_bw.replace('.', '_')}"
  if pcie_bw != DEFAULT_PCIE_BW:
    characteristics_str += f"-pcie{pcie_bw.replace('.', '_')}"
  if kernel_time_variation != DEFAULT_KERNEL_TIME_VARIATION:
    characteristics_str += f"-var{kernel_time_variation.replace('.', '_')}"
  if kernel_speedup_variation != DEFAULT_KERNEL_SPEEDUP_VARIATION:
    characteristics_str += f"-spd{kernel_speedup_variation.replace('.', '_')}"

  config_name = f"{config_basefolder}/{config_foldername}/{batch_size}-sim_{policy}{characteristics_str}.config"
  config_output_folder = f"../results/{output_foldername}/{batch_size}-{policy}{characteristics_str}"

  assert config_name not in config_names, f"{config_name} already exists"
  assert config_output_folder not in config_output_folders, f"{config_output_folder} already exists"
  config_names.add(config_name)
  config_output_folders.add(config_output_folder)
  global num_run_configs
  num_run_configs += 1

  mig_policy = "DEEPUM" if policy.startswith("deepUM") else \
               "FLASHNEURON" if policy.startswith("FlashNeuron") else \
               "G10GDSSSD" if policy.startswith("G10GDSSSD") else \
               "G10GDSFULL" if policy.startswith("G10GDSFULL") else \
               ""
  return [f"""output_folder           {config_output_folder}
is_simulation           1

{"is_transformer          1" if model in ["BERT", "VIT", "NeRF"] else ""}
{"trans_borden            210" if model == "BERT" else "trans_borden            184" if model == "VIT" else "trans_borden            7" if model == "NeRF" else ""}
{"is_inception            1" if model == "Inceptionv3" else  "is_resnet               1" if model == "ResNet152" or model == "ResNeXt101_32" or model == "WResNet101" else "is_senet                1"}
batch_size              {batch_size}
input_H                 {299 if model == "Inceptionv3" else 224}
input_W                 {299 if model == "Inceptionv3" else 224}
num_iteration           2
num_threads             128

nn_model_input_file     ../frontend/Nets/{netname}.txt
orig_kernel_time_file   ../results/{output_foldername}/cudnn{batch_size}.txt
pf_kernel_time_file     ../results/{output_foldername}/cudnn{batch_size}PF.txt
input_pf_kernel_time_file ../results/{output_foldername}/cudnn{batch_size}InputPF.txt
workspace_size_file       ../results/{output_foldername}/cudnn{batch_size}Workspace.txt
stat_output_file        sim_result
is_UVM                  1
use_prefetch            {"1" if policy.startswith("prefetch") or policy.startswith("deepUM") or policy.startswith("FlashNeuron") else "0"}
eviction_policy         {policy.split("_")[-1].upper() if policy.split("_")[-1].upper() != "FLASHNEURON" and policy.split("_")[-1].upper().find("G10GDS") == -1 else "LRU"}
{f"migration_policy        {mig_policy}" if mig_policy != "" else ""}
{"prefetch_degree         8" if policy.startswith("deepUM") else ""}
system_latency_us       45

CPU_PCIe_bandwidth_GBps {pcie_bw}
CPU_memory_line_GB      {hostmem_size if policy != "G10GDSSSD" else 0}

GPU_memory_size_GB      40
GPU_frequency_GHz       1.2
GPU_PCIe_bandwidth_GBps {pcie_bw}
GPU_malloc_uspB         0.000000814
GPU_free_uspB           0

SSD_PCIe_bandwidth_GBps {ssd_bw}
SSD_read_latency_us     12
SSD_write_latency_us    16
SSD_latency_us          20

PCIe_latency_us         5
PCIe_batch_size_page    50

delta_parameter         0.5
{f"kernel_time_std_dev     {kernel_time_variation}" if kernel_time_variation != "0" else ""}
{f"ran_seed                {'15' if model == 'BERT' else '10' if model == 'ResNet152' else '25' if model == 'SENet154' else '1'}" if kernel_time_variation != "0" else ""}
""", config_name]


PF_MODE = Enum("PF_MODE", ["NO_PF", "INPUT_PF", "OUTPUT_PF", "PF"])
def get_profile_config_specification(model: str, netname: str, config_foldername: str, output_foldername: str, batch_size: str, pf_mode: Enum, is_individual: bool=False, is_cudnn: bool=True):
  pf_string = "-InputPF" if pf_mode == PF_MODE.INPUT_PF else "-OutputPF" if pf_mode == PF_MODE.OUTPUT_PF else "-PF" if pf_mode == PF_MODE.PF else ""

  config_name = f"{config_basefolder}/{config_foldername}/{batch_size}-profile{pf_string}.config"
  config_output_folder = f"""{output_foldername}/profile-{batch_size}-{"cudnn" if is_cudnn else "manual"}{pf_string}"""

  assert config_name not in config_names, f"{config_name} already exists"
  assert config_output_folder not in config_output_folders, f"{config_output_folder} already exists"
  config_names.add(config_name)
  config_output_folders.add(config_output_folder)
  global num_profile_configs
  num_profile_configs += 1

  return [f"""output_folder           {config_output_folder}
is_profiling            1
is_individual           {"1" if is_individual else "0"}
is_cudnn                {"1" if is_cudnn else "0"}
is_UVM                  {"0" if pf_mode == PF_MODE.NO_PF else "1"}
is_input_pf_only        {"1" if pf_mode == PF_MODE.INPUT_PF else "0"}
is_output_pf_only       {"1" if pf_mode == PF_MODE.OUTPUT_PF else "0"}

{"is_transformer          1" if model in ["BERT", "VIT", "NeRF"] else ""}
{"trans_borden            210" if model == "BERT" else "trans_borden            184" if model == "VIT" else "trans_borden            7" if model == "NeRF" else ""}
{"is_inception            1" if model == "Inceptionv3" else  "is_resnet               1" if model == "ResNet152" or model == "ResNeXt101_32" or model == "WResNet101" else "is_senet                1"}
batch_size              {batch_size}
input_H                 {299 if model == "Inceptionv3" else 224}
input_W                 {299 if model == "Inceptionv3" else 224}

nn_model_input_file     ../frontend/Nets/{netname}.txt
""", config_name]

def profiling():
  for model, configs in models.items():
    netname = configs[0]
    config_foldername = configs[1]
    output_foldername = configs[2]
    batch_sizes = configs[3]
    try:
      os.mkdir(f"{config_basefolder}/{config_foldername}")
    except OSError:
      pass
    for pf_mode in [PF_MODE.NO_PF, PF_MODE.INPUT_PF, PF_MODE.PF]:
      for batch_size in batch_sizes:
        content, filename = get_profile_config_specification(model, netname, config_foldername, output_foldername, batch_size, pf_mode)
        with open(filename, "w") as f:
          f.write(content)

def main_performance():
  for model, configs in models.items():
    netname = configs[0]
    config_foldername = configs[1]
    output_foldername = configs[2]
    batch_sizes = configs[3]
    try:
      os.mkdir(f"{config_basefolder}/{config_foldername}")
    except OSError:
      pass
    for policy in policies:
      for batch_size in batch_sizes:
        content, filename = get_run_config_specification(model, netname, config_foldername, output_foldername, policy, batch_size)
        with open(filename, "w") as f:
          f.write(content)

def cpu_varying():
  for model, configs in models.items():
    netname = configs[0]
    config_foldername = configs[1]
    output_foldername = configs[2]
    batch_sizes = configs[3]
    try:
      os.mkdir(f"{config_basefolder}/{config_foldername}")
    except OSError:
      pass
    for policy in cpu_policies:
      for batch_size in batch_sizes:
        if model.find("ResNet") != -1:  
          temp_hostmem_sizes = [*hostmem_sizes, "192"]
        elif model.find("SENet") != -1:  
          temp_hostmem_sizes = [*hostmem_sizes, "96"]
        else:
          temp_hostmem_sizes = hostmem_sizes
        for hostmem_size in temp_hostmem_sizes:
          content, filename = get_run_config_specification(model, netname, config_foldername, output_foldername, policy, batch_size, hostmem_size=hostmem_size)
          with open(filename, "w") as f:
            f.write(content)

def pcie_varying():
  for model, configs in models.items():
    netname = configs[0]
    config_foldername = configs[1]
    output_foldername = configs[2]
    batch_sizes = configs[3]
    try:
      os.mkdir(f"{config_basefolder}/{config_foldername}")
    except OSError:
      pass
    for policy in cpu_policies:
      for batch_size in batch_sizes:
          for pcie_bw in pcie_bws:
            content, filename = get_run_config_specification(model, netname, config_foldername, output_foldername, policy, batch_size, pcie_bw=pcie_bw)
            with open(filename, "w") as f:
              f.write(content)

def ssd_varying():
  for model, configs in models.items():
    netname = configs[0]
    config_foldername = configs[1]
    output_foldername = configs[2]
    batch_sizes = configs[3]
    try:
      os.mkdir(f"{config_basefolder}/{config_foldername}")
    except OSError:
      pass
    for policy in ssd_policies:
      for batch_size in batch_sizes:
          for ssd_bw in ssd_bws:
            pcie_bw = "32" if float(ssd_bw) < 32 else ssd_bw
            content, filename = get_run_config_specification(model, netname, config_foldername, output_foldername, policy, batch_size, pcie_bw=pcie_bw, ssd_bw=ssd_bw)
            with open(filename, "w") as f:
              f.write(content)
              
def kernel_time_varying():
  for model, configs in models.items():
    netname = configs[0]
    config_foldername = configs[1]
    output_foldername = configs[2]
    batch_sizes = configs[3]
    try:
      os.mkdir(f"{config_basefolder}/{config_foldername}")
    except OSError:
      pass
    for batch_size in batch_sizes:
      for kernel_time_variation in kernel_time_variations:
        content, filename = get_run_config_specification(model, netname, config_foldername, output_foldername, "prefetch_lru", batch_size, kernel_time_variation=kernel_time_variation)
        with open(filename, "w") as f:
          f.write(content)
              
def kernel_speedup_varying():
  for model, configs in models.items():
    netname = configs[0]
    config_foldername = configs[1]
    output_foldername = configs[2]
    batch_sizes = configs[3]
    try:
      os.mkdir(f"{config_basefolder}/{config_foldername}")
    except OSError:
      pass
    for policy in kernel_speedup_policies:
      for batch_size in batch_sizes:
        for kernel_speedup_variation in kernel_speedup_variations:
          content, filename = get_run_config_specification(model, netname, config_foldername, output_foldername, policy, batch_size, hostmem_size="128", pcie_bw="32", ssd_bw="6.4", kernel_speedup_variation=kernel_speedup_variation)
          with open(filename, "w") as f:
            f.write(content)


print(f"Configs are generated to folder <{config_basefolder}>")
profiling()
ssd_varying()
# pcie_varying()
cpu_varying()
kernel_time_varying()
# kernel_speedup_varying()
main_performance()
print(f"Total {len(config_output_folders)} configs generated ({num_run_configs} run config + {num_profile_configs} profile config)")
