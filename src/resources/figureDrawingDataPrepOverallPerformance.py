import os, json, codecs
import numpy as np
from statsFiguresUtil import *

script_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.abspath(os.path.join(script_path, os.path.pardir, "configs"))
result_path = os.path.abspath(os.path.join(script_path, os.path.pardir, os.path.pardir, "results"))
output_path = os.path.abspath(os.path.join(script_path, "raw_output"))

def retrieve_data_from_json_file(filename: str):
  overall_json = json.load(codecs.open(filename, 'r', encoding='utf-8'))
  dimension_num = overall_json["dimension_num"]
  dimension_names = overall_json["dimension_names"]
  dimension_details = []
  print(f"Parsing json file <{filename}>")
  for dimension in dimension_names:
    exec(f"{dimension}s = overall_json[\"{dimension}\"]")
    exec(f"print(f\"  {dimension:12s}: {{{dimension}s}}\")")
    exec(f"dimension_details.append({dimension}s)")
  data = np.array(overall_json["data"])
  print(f"  Dimension Length: {[len(dimension_detail) for dimension_detail in dimension_details]}")
  return dimension_num, dimension_names, dimension_details, data

def fuse_data_matrices(items: list):
  assert all(item[0] == len(all_dimensions) for item in items)
  assert all(item[1] == list(all_dimensions.keys()) for item in items)
  for dimension in all_dimensions.keys():
    exec(f"{dimension}s_set = set()")
  # gather all the properties in all dimensions into a set
  for item in items:
    for dimension_idx, dimension in enumerate(all_dimensions.keys()):
      exec(f"for property in item[2][dimension_idx].keys(): {dimension}s_set.add(property)")
  # sorting keys for all dimension according to lambda expression
  print("===========================================")
  print(">>> Unified Dimensions:")
  dimension_details = []
  for dimension in all_dimensions.keys():
    exec(f"global {dimension}s; {dimension}s = {{ key : i for i, key in enumerate(sorted(list({dimension}s_set), key=all_dimensions[\"{dimension}\"])) }}")
    exec(f"print(f\"  {dimension:12s}: {{{dimension}s}}\")")
    exec(f"dimension_details.append(list({dimension}s.keys()))")
  # generate unified data matrix
  data, data_dimension = -np.ones(0, dtype=float), []
  for dimension in all_dimensions.keys():
    exec(f"data_dimension.append(str(len({dimension}s)))")
  # python3 does not support exec to directly reassign local variable
  exec(f"data.resize(({', '.join(data_dimension)}), refcheck=False)")
  exec(f"data.fill(-1)")
  data_dimension = tuple(int(dimension_len) for dimension_len in data_dimension)
  print(f"  Dimension Length: {list(data_dimension)}")
  print("===========================================")
  for data_idx in np.ndindex(data_dimension):
    properties = [dimension_details[i][data_idx[i]] for i in range(len(all_dimensions))]
    for item_idx, item in enumerate(items):
      _, _, item_dimension_details, item_data = item
      if all([properties[i] in item_dimension_details[i] for i in range(len(all_dimensions))]):
        item_data_index = [item_dimension_details[i][properties[i]] for i in range(len(all_dimensions))]
        data[data_idx] = max(data[data_idx], item_data[tuple(item_data_index)])
  return data

if __name__ == "__main__":
  data_used = ["data"]
  # data_used = ["data100copy", "data113copy"]
  data = fuse_data_matrices([
      retrieve_data_from_json_file(f"{output_path}/{data_json}.json")
      for data_json in data_used])

  header_printed = False
  with open(f"figure_drawing/overall_performance/all.base.txt", "w") as f:
    for model in [BERT, VIT, INCEPTION, RESNET, SENET]:
      # settings BEGIN ========================================
      # model = INCEPTION
      # model = RESNET
      # model = SENET
      # model = RESNEXT
      # model = VIT
      # model = BERT

      stat_candidate = "exe_time"
      # stat_candidate = "pf_num"

      # transpose = True
      transpose = False 

      # plot_format = False
      # latex_format = True
      # plot_format = True
      # latex_format = False
      plot_format = True
      latex_format = False

      include_ideal = True
      normal_have_axis = True

      # unit = ""
      unit = "k"
      # unit = "M"
      
      if model == INCEPTION:
        # Inception ###################################
        x_axis_idxs, y_axis_idxs = [
          tuple([
              batch_sizes["1536"]
          ]),
          # tuple(
          #     batch_sizes.values()
          # ),
          tuple([
              settings["lru"],
              settings["FlashNeuron"], 
              settings["deepUM"],
              settings["G10GDSSSD"],
              settings["G10GDSFULL"], 
              settings["prefetch_lru"]
          ])
        ]
      elif model == RESNET:
        # ResNet ######################################
        x_axis_idxs, y_axis_idxs = [
          tuple([
              batch_sizes["1280"]
          ]),
          # tuple(
          #     batch_sizes.values()
          # ),
          tuple([
              settings["lru"],
              settings["FlashNeuron"], 
              settings["deepUM"],
              settings["G10GDSSSD"],
              settings["G10GDSFULL"], 
              settings["prefetch_lru"]
          ])
        ]
      elif model == SENET:
        # SENet #######################################
        x_axis_idxs, y_axis_idxs = [
          tuple([
              batch_sizes["1024"]
          ]),
          tuple([
              settings["lru"],
              settings["FlashNeuron"], 
              settings["deepUM"],
              settings["G10GDSSSD"],
              settings["G10GDSFULL"], 
              settings["prefetch_lru"]
          ])
        ]
      elif model == BERT:
        # BERT ########################################
        x_axis_idxs, y_axis_idxs = [
          tuple([
              batch_sizes["256"]
          ]),
          # tuple(
          #     batch_sizes.values()
          # ),
          tuple([
              settings["lru"],
              settings["FlashNeuron"], 
              settings["deepUM"],
              settings["G10GDSSSD"],
              settings["G10GDSFULL"], 
              settings["prefetch_lru"]
          ])
        ]
      elif model == VIT:
        # VIT #########################################
        x_axis_idxs, y_axis_idxs = [
          tuple([
              batch_sizes["1280"]
          ]),
          # tuple(
          #     batch_sizes.values()
          # ),
          tuple([
              settings["lru"],
              settings["FlashNeuron"], 
              settings["deepUM"],
              settings["G10GDSSSD"],
              settings["G10GDSFULL"], 
              settings["prefetch_lru"]
          ])
        ]
      elif model == RESNEXT:
        # ResNeXt #####################################
        x_axis_idxs, y_axis_idxs = [
          tuple([
              batch_sizes["256"], 
              batch_sizes["512"], 
              batch_sizes["768"], 
              batch_sizes["1024"]
          ]),
          # tuple(
          #     batch_sizes.values()
          # ),
          tuple([
              settings["lru"],
              settings["FlashNeuron"], 
              settings["deepUM"],
              # settings["G10GDSSSD"],
              # settings["G10GDSFULL"], 
              settings["prefetch_lru"]
          ])
        ]
      ###############################################
      # settings END ==========================================

      # sanity check
      assert plot_format + latex_format <= 1
      assert stat_candidate in stats

      # data transformation & auto generation
      model_desc = list(net_name_detail_translation.values())[model]
      model = list(net_name_translation.values())[model]
      print(f"Model being processed: {model}")
      if stat_candidate == "pf_num":
        include_ideal = False
        y_axis_idxs = [y_axis_idx for y_axis_idx in y_axis_idxs if y_axis_idx != settings["FlashNeuron"]]
      if not plot_format:
        x_ticks, y_ticks = [
          [list(batch_sizes.keys())[i] for i in x_axis_idxs],
          [setting_translation[list(settings.keys())[i]] for i in y_axis_idxs]
        ]
      else:
        x_ticks, y_ticks = [
          [list(batch_sizes.keys())[i] for i in x_axis_idxs],
          [list(settings.keys())[i] for i in y_axis_idxs]
        ]
      print(f"Index selection:\n  X:{x_ticks}\n  Y:{y_ticks}")

      data_slice = data[
          models[model], 
          x_axis_idxs, 
          :, 
          cpu_mems["128"], 
          ssd_bws["3.2"],
          pcie_bws["15.754"],
          ktime_vars["0"],
          stats[stat_candidate]
      ].astype(int)
      data_slice = data_slice[:, y_axis_idxs]

      if include_ideal:
        data_slice_ideal = data[
            models[model], 
            x_axis_idxs, 
            settings["lru"], 
            cpu_mems["128"], 
            ssd_bws["3.2"],
            pcie_bws["15.754"],
            ktime_vars["0"],
            stats["ideal_exe_time"]
        ].astype(int)
        data_slice_ideal = data_slice_ideal[..., np.newaxis]
        
        data_slice = np.hstack((data_slice, data_slice_ideal))
        y_ticks.append("ideal")

      if transpose:
        x_ticks, y_ticks = y_ticks, x_ticks
        x_axis_idxs, y_axis_idxs = y_axis_idxs, x_axis_idxs
        data_slice = data_slice.T
      print(f"Slice shape: {data_slice.shape}")

      print(f"Slice content:")
      dividend = 1000 if unit == "k" else 1000000 if unit == "M" else 1
      slice_content = ""
      if not header_printed:
        f.write(" | ".join(y_ticks) + "\n\n")
        header_printed = True
      f.write(model_desc + "\n")
      if latex_format:
        slice_content += "TABLE & "
        slice_content += " & ".join(y_ticks) + " \\\\\\hline\n"
        for x_idx, x_tick in enumerate(x_ticks):
          slice_content += x_tick + " & "
          slice_content += " & ".join(f"{d if d < dividend else int(d / dividend)}{'' if d < dividend else unit}" for d in data_slice[x_idx, :]) + " \\\\\\hline\n"
      elif plot_format:
        slice_content += " | ".join(y_ticks) + "\n\n"
        for x_idx, x_tick in enumerate(x_ticks):
          slice_content += x_tick + "\n"
          slice_content += " ".join(str(d) if int(d) != -1 else "inf" for d in data_slice[x_idx, :]) + "\n\n"
          f.write(" ".join(str(d) if int(d) != -1 else "inf" for d in data_slice[x_idx, :]) + "\n\n")
      else:
        if normal_have_axis:
          slice_content += " " * 15
          for j in range(data_slice.shape[1]):
            slice_content += f"{y_ticks[j]:>15s}"
          slice_content += "\n"
        for i in range(data_slice.shape[0]):
          if normal_have_axis:
            slice_content += f"{x_ticks[i]:>15s}"
          for j in range(data_slice.shape[1]):
            slice_content += f"{data_slice[i, j]:15d}"
          slice_content += "\n"
      print(slice_content)
