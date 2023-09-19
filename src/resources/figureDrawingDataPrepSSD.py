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
  print("Unified Dimensions:")
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

  for model in [INCEPTION, RESNET, SENET, VIT, BERT]:
    # settings BEGIN ========================================

    stat_candidate = "exe_time"

    transpose = False

    plot_format = True
    latex_format = False

    include_ideal = True
    normal_have_axis = True

    unit = "k"

    if model == INCEPTION:
      # Inception ###################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            # ssd_bws["3.2"],
            ssd_bws["6.4"],
            ssd_bws["12.8"],
            ssd_bws["19.2"],
            ssd_bws["25.6"],
            ssd_bws["32"]
        ]),
        tuple([
            settings["lru"],
            settings["FlashNeuron"],
            settings["deepUM"],
            settings["prefetch_lru"]
        ])
      ]
      batch_size = "1536"
    elif model == RESNET:
      # ResNet ######################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            # ssd_bws["3.2"],
            ssd_bws["6.4"],
            ssd_bws["12.8"],
            ssd_bws["19.2"],
            ssd_bws["25.6"],
            ssd_bws["32"]
        ]),
        # tuple(
        #     batch_sizes.values()
        # ),
        tuple([
            settings["lru"],
            settings["FlashNeuron"],
            settings["deepUM"],
            settings["prefetch_lru"]
        ])
      ]
      batch_size = "1280"
    elif model == SENET:
      # SENet #######################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            # ssd_bws["3.2"],
            ssd_bws["6.4"],
            ssd_bws["12.8"],
            ssd_bws["19.2"],
            ssd_bws["25.6"],
            ssd_bws["32"]
        ]),
        tuple([
            settings["lru"],
            settings["FlashNeuron"],
            settings["deepUM"],
            settings["prefetch_lru"]
        ])
      ]
      batch_size = "1024"
    elif model == BERT:
      # BERT ########################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            # ssd_bws["3.2"],
            ssd_bws["6.4"],
            ssd_bws["12.8"],
            ssd_bws["19.2"],
            ssd_bws["25.6"],
            ssd_bws["32"]
        ]),
        tuple([
            settings["lru"],
            settings["FlashNeuron"],
            settings["deepUM"],
            settings["prefetch_lru"]
        ])
      ]
      batch_size = "512"
    elif model == VIT:
      # VIT #########################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            # ssd_bws["3.2"],
            ssd_bws["6.4"],
            ssd_bws["12.8"],
            ssd_bws["19.2"],
            ssd_bws["25.6"],
            ssd_bws["32"]
        ]),
        tuple([
            settings["lru"],
            settings["FlashNeuron"],
            settings["deepUM"],
            settings["prefetch_lru"]
        ])
      ]
      batch_size = "1280"
    ###############################################
    # settings END ==========================================

    # sanity check
    assert plot_format + latex_format <= 1
    assert stat_candidate in stats

    # data transformation & auto generation
    out_filename = list(net_name_translation.keys())[model]
    model = list(net_name_translation.values())[model]
    print(f"Model being processed: {model}")
    if stat_candidate == "pf_num":
      include_ideal = False
      y_axis_idxs = [y_axis_idx for y_axis_idx in y_axis_idxs if y_axis_idx != settings["FlashNeuron"]]
    if not plot_format:
      x_ticks, y_ticks = [
        [list(ssd_bws.keys())[i] for i in x_axis_idxs],
        [setting_translation[list(settings.keys())[i]] for i in y_axis_idxs]
      ]
    else:
      x_ticks, y_ticks = [
        [list(ssd_bws.keys())[i] for i in x_axis_idxs],
        [list(settings.keys())[i] for i in y_axis_idxs]
      ]

    data_slice = data[
        models[model],
        batch_sizes[batch_size],
        y_axis_idxs,
        cpu_mems["128"],
        :,
        pcie_bws["32"],
        ktime_vars["0"],
        stats[stat_candidate]
    ].astype(int)
    data_slice = data_slice[:, x_axis_idxs].T

    if include_ideal:
      data_slice_ideal = data[
          models[model],
          batch_sizes[batch_size],
          settings["lru"],
          cpu_mems["128"],
          x_axis_idxs,
          pcie_bws["32"],
          ktime_vars["0"],
          stats["ideal_exe_time"]
      ].astype(int)
      data_slice_ideal = data_slice_ideal[..., np.newaxis]

      data_slice = np.hstack((data_slice, data_slice_ideal))
      y_ticks.append("ideal")
    print(f"Index selection:\n  X:{x_ticks}\n  Y:{y_ticks}")

    if transpose:
      x_ticks, y_ticks = y_ticks, x_ticks
      x_axis_idxs, y_axis_idxs = y_axis_idxs, x_axis_idxs
      data_slice = data_slice.T
    print(f"Slice shape: {data_slice.shape}")

    print(f"Slice content:")
    dividend = 1000 if unit == "k" else 1000000 if unit == "M" else 1
    slice_content = ""
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
      with open(f"figure_drawing/sensitivity_ssdbw/{out_filename}.txt", "w") as f:
        f.write(slice_content)
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
