import os, subprocess, json, codecs
import numpy as np
from statsFiguresUtil import *

legacy_support = True

script_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.abspath(os.path.join(script_path, os.path.pardir, "configs"))
result_path = os.path.abspath(os.path.join(script_path, os.path.pardir, os.path.pardir, "results"))
output_path = os.path.abspath(os.path.join(script_path, "raw_output"))
output_json_file = f"{output_path}/data.json"

if __name__ == "__main__":
  stat_to_gather = [
    "ideal_exe_time", 
    "exe_time", 
    "cpu_pf", 
    "ssd_pf", 
    "unalloc_pf", 
    "total_evc", 
    "incoming_pg_cpu", 
    "incoming_pg_ssd", 
    "outgoing_pg_cpu", 
    "outgoing_pg_ssd",
    "total_time_breakdown_stall",
    "total_time_breakdown_overlap",
    "total_time_breakdown_executiuonOnly"
  ]
  stat_to_gather = { stat_name : [np.inf, np.inf] for stat_name in stat_to_gather }
  post_processed_stats_dict = { stat_name : 0 for stat_name in post_processed_stats }

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  for dimension in all_dimensions.keys():
    exec(f"{dimension}s = dict()")

  # dimension and name pass
  final_stats, err = subprocess.Popen(
      ["find", result_path, "-name", "*.final", "-type", "f"],
      stdout=subprocess.PIPE).communicate()
  final_stats = final_stats.decode().strip().split("\n")
  for final_stat in final_stats:
    model = os.path.basename(os.path.dirname(os.path.dirname(final_stat)))
    config = os.path.basename(os.path.dirname(final_stat))
    config_split = config.split("-")
    assert len(config_split) >= 2
    model_config = f"{model}.{config_split[0]}"
    experiment_config = config_split[1]
    line, err = subprocess.Popen(
        ["tail", "-1", final_stat],
        stdout=subprocess.PIPE).communicate()
    if line.decode().strip() != "-1" and experiment_config.lower().find("flash") == -1 and experiment_config.lower().find("g10gds") == -1:
      continue
    # processing characteristics of each file
    batch_size, setting = config_split[:2]
    cpu_mem, ssd_bw, pcie_bw, ktime_var = "128", "3.2", "15.754", "0"
    characteristics = config_split[2:]
    for characteristic in characteristics:
      if characteristic.startswith("cpu"):
        cpu_mem = characteristic[3:].replace("_", ".")
      elif characteristic.startswith("ssd"):
        ssd_bw = characteristic[3:].replace("_", ".")
      elif characteristic.startswith("pcie"):
        pcie_bw = characteristic[4:].replace("_", ".")
      elif characteristic.startswith("var"):
        ktime_var = characteristic[3:].replace("_", ".")
    for stat in post_processed_stats:
      # loading data into data array
      for dimension in all_dimensions.keys():
        exec(f"if {dimension} not in {dimension}s: {dimension}s[{dimension}] = len({dimension}s)")
        
  # sorting keys for all dimension according to lambda expression
  for dimension in all_dimensions.keys():
    exec(f"{dimension}s = {{ key : i for i, key in enumerate(sorted({dimension}s.keys(), key=all_dimensions[\"{dimension}\"])) }}")
    exec(f"print(f\"{dimension:12s}: {{{dimension}s}}\")")
  # initialize data array according to dimension
  data, data_dimension = np.zeros(0), []
  for dimension in all_dimensions.keys():
    exec(f"data_dimension.append(str(len({dimension}s)))")
  exec(f"data = -np.ones(({', '.join(data_dimension)}), dtype=float)")
  # initialize output json structure
  overall_json = { "dimension_num" : len(all_dimensions), "dimension_names" : list(all_dimensions.keys()) }
  for dimension in all_dimensions.keys():
    exec(f"overall_json[dimension] = {dimension}s")
  
  # data pass
  for final_stat in final_stats:
    model = os.path.basename(os.path.dirname(os.path.dirname(final_stat)))
    config = os.path.basename(os.path.dirname(final_stat))
    config_split = config.split("-")
    assert len(config_split) >= 2
    model_config = f"{model}.{config_split[0]}"
    experiment_config = config_split[1]
    line, err = subprocess.Popen(
        ["tail", "-1", final_stat],
        stdout=subprocess.PIPE).communicate()
    if line.decode().strip() != "-1" and experiment_config.lower().find("flash") == -1 and experiment_config.lower().find("g10gds") == -1:
      continue
    # processing characteristics of each file
    batch_size, setting = config_split[:2]
    cpu_mem, ssd_bw, pcie_bw, ktime_var = "128", "3.2", "15.754", "0"
    characteristics = config_split[2:]
    for characteristic in characteristics:
      if characteristic.startswith("cpu"):
        cpu_mem = characteristic[3:].replace("_", ".")
      elif characteristic.startswith("ssd"):
        ssd_bw = characteristic[3:].replace("_", ".")
      elif characteristic.startswith("pcie"):
        pcie_bw = characteristic[4:].replace("_", ".")
      elif characteristic.startswith("var"):
        ktime_var = characteristic[3:].replace("_", ".")
    
    # init before every gather
    for key in stat_to_gather.keys():
      stat_to_gather[key] = [np.inf, np.inf]
    # read the file and parse results
    # TODO: merge this loop with pevious one
    with open(final_stat, "r") as f:
      for line in f.readlines():
        if line.find("iter0") != -1:
          # Other configs
          line_split = line.split("=")
          if len(line_split) != 2:
            continue
          stat_name, stat_val = line_split
          stat_split = stat_name.strip().split(".")
          stat_val = stat_val.strip()
          assert len(stat_split) == 3 or len(stat_split) == 2
          if len(stat_split) == 3:
            stat_name = f"{stat_split[2].strip()}"
            if stat_name in stat_to_gather:
              stat_to_gather[stat_name][0] = int(stat_val)
          elif len(stat_split) == 2:
            stat_name = f"{stat_split[0].strip()}"
            if stat_name in stat_to_gather:
              stat_to_gather[stat_name][0] = int(stat_val)
        elif line.find("iter1") != -1:
          # Other configs
          line_split = line.split("=")
          if len(line_split) != 2:
            continue
          stat_name, stat_val = line_split
          stat_split = stat_name.strip().split(".")
          stat_val = stat_val.strip()
          assert len(stat_split) == 3 or len(stat_split) == 2
          if len(stat_split) == 3:
            stat_name = f"{stat_split[2].strip()}"
            if stat_name in stat_to_gather:
              stat_to_gather[stat_name][1] = int(stat_val)
          elif len(stat_split) == 2:
            stat_name = f"{stat_split[0].strip()}"
            if stat_name in stat_to_gather:
              stat_to_gather[stat_name][1] = int(stat_val)
        elif line.find("total_exe_time") != -1:
          # FlashNeuron, G10GDSSSD specific
          line_split = line.split("=")
          if len(line_split) != 2:
            continue
          stat_to_gather['exe_time'][0] = int(float(line_split[1]) / 1000 * 1200 * 10 ** 6)
        elif line.find("total_time_breakdown_stall") != -1:
          # FlashNeuron, G10GDSSSD specific
          line_split = line.split("=")
          if len(line_split) != 2:
            continue
          stat_to_gather['total_time_breakdown_stall'][0] = float(line_split[1])
        elif line.find("total_time_breakdown_overlap") != -1:
          # FlashNeuron, G10GDSSSD specific
          line_split = line.split("=")
          if len(line_split) != 2:
            continue
          stat_to_gather['total_time_breakdown_overlap'][0] = float(line_split[1])
        elif line.find("total_time_breakdown_executionOnly") != -1:
          # FlashNeuron, G10GDSSSD specific
          line_split = line.split("=")
          if len(line_split) != 2:
            continue
          stat_to_gather['total_time_breakdown_executiuonOnly'][0] = float(line_split[1])
        elif line.find("total_ssd2gpu_byte") != -1:
          # FlashNeuron, G10GDSSSD specific
          line_split = line.split("=")
          if len(line_split) != 2:
            continue
          pg_num = float(line_split[1]) / 4096
          if stat_to_gather['incoming_pg_ssd'][0] == np.inf:
            stat_to_gather['incoming_pg_ssd'][0] = pg_num
          else:
            stat_to_gather['incoming_pg_ssd'][0] += pg_num
          stat_to_gather['incoming_pg_cpu'][0] = 0
        elif line.find("total_gpu2ssd_byte") != -1:
          # FlashNeuron specific
          line_split = line.split("=")
          if len(line_split) != 2:
            continue
          pg_num = float(line_split[1]) / 4096
          if stat_to_gather['outgoing_pg_ssd'][0] == np.inf:
            stat_to_gather['outgoing_pg_ssd'][0] = pg_num
          else:
            stat_to_gather['outgoing_pg_ssd'][0] += pg_num
          stat_to_gather['outgoing_pg_cpu'][0] = 0
          

    select_idx = 0 if stat_to_gather["exe_time"][0] < stat_to_gather["exe_time"][1] else 1
    assert stat_to_gather["total_time_breakdown_stall"][select_idx] >= 0
    assert stat_to_gather["total_time_breakdown_overlap"][select_idx] >= 0
    assert stat_to_gather["total_time_breakdown_executiuonOnly"][select_idx] >= 0
    if setting.strip() == "lru":
      stat_to_gather["total_time_breakdown_stall"][select_idx] -= stat_to_gather["total_time_breakdown_overlap"][select_idx]
      stat_to_gather["total_time_breakdown_overlap"][select_idx] = 0
    total_time_breakdown = stat_to_gather["total_time_breakdown_stall"][select_idx] + \
                           stat_to_gather["total_time_breakdown_overlap"][select_idx] + \
                           stat_to_gather["total_time_breakdown_executiuonOnly"][select_idx]
    assert total_time_breakdown != 0
    post_processed_stats_dict["ideal_exe_time"] = stat_to_gather["ideal_exe_time"][select_idx]
    post_processed_stats_dict["exe_time"] = stat_to_gather["exe_time"][select_idx]
    post_processed_stats_dict["pf_num"] = stat_to_gather['cpu_pf'][select_idx] + \
                                          stat_to_gather['ssd_pf'][select_idx] + \
                                          stat_to_gather['unalloc_pf'][select_idx]
    post_processed_stats_dict["ssd2gpu_traffic"] = stat_to_gather["incoming_pg_ssd"][select_idx]
    post_processed_stats_dict["gpu2ssd_traffic"] = stat_to_gather["outgoing_pg_ssd"][select_idx]
    post_processed_stats_dict["cpu2gpu_traffic"] = stat_to_gather["incoming_pg_cpu"][select_idx]
    post_processed_stats_dict["gpu2cpu_traffic"] = stat_to_gather["outgoing_pg_cpu"][select_idx]
    post_processed_stats_dict["stall_percentage"] = stat_to_gather["total_time_breakdown_stall"][select_idx]
    post_processed_stats_dict["overlap_percentage"] = stat_to_gather["total_time_breakdown_overlap"][select_idx]
    post_processed_stats_dict["compute_percentage"] = stat_to_gather["total_time_breakdown_executiuonOnly"][select_idx]
    for stat, val in post_processed_stats_dict.items():
      # write to final data
      index_comp = []
      index_str = ""
      for dimension in all_dimensions:
        exec(f"index_comp.append(str({dimension}s[{dimension}]))")
      # print(f"{model}-{config:27s}", model, batch_size, setting, cpu_mem, ssd_bw, pcie_bw, stat, index_comp, val)
      exec(f"data[{', '.join(index_comp)}] = val")
  # write data back to json
  overall_json["data"] = data.tolist()
  json.dump(overall_json, codecs.open(output_json_file, 'w', encoding='utf-8'), 
            separators=(',', ':'), 
            sort_keys=False, 
            indent=2)
  print(f"All data is saved to <{output_json_file}>")

