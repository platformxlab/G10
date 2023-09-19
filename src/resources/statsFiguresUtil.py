INCEPTION, RESNET, SENET, BERT, VIT, RESNEXT = range(6)

# define all dimensions script need to gather, and their corresponding sording rules
post_processed_stats = [
  "ideal_exe_time", 
  "exe_time", 
  "pf_num", 
  "evc_num",
  "ssd2gpu_traffic",
  "gpu2ssd_traffic",
  "cpu2gpu_traffic",
  "gpu2cpu_traffic",
  "stall_percentage",
  "overlap_percentage",
  "compute_percentage"
]

all_dimensions = { 
  "model":      lambda x : ["resnet152", "resnext101_32", "senet154", "inceptionv3", "bert_base", "vit"].index(x.lower()), 
  "batch_size": lambda x : float(x), 
  "setting":    lambda x : ["lru", "flashneuron", "deepum", "prefetch_lru", "g10gdsssd", "g10gdsfull"].index(x.lower()), 
  "cpu_mem":    lambda x : float(x), 
  "ssd_bw":     lambda x : float(x), 
  "pcie_bw":    lambda x : float(x),
  "ktime_var":  lambda x : float(x),
  "stat":       lambda x : post_processed_stats.index(x.lower())
}

setting_translation = {
  "lru"          : "Base UVM",
  "FlashNeuron"  : "FlashNeuron",
  "deepUM"       : "DeepUM+",
  "prefetch_lru" : "G10",
  "G10GDSSSD"    : "G10-GDS",
  "G10GDSFULL"   : "G10-Host",
  "ideal"        : "Ideal"
}

x_label_translation = {
  "256"  : "B = 256\nM = NaN%",
  "512"  : "B = 512\nM = NaN%",
  "768"  : "B = 768\nM = NaN%",
  "1024" : "B = 1024\nM = NaN%",
  "1280" : "B = 1280\nM = NaN%",
  "1536" : "B = 1536\nM = NaN%"
}

# keep this EXACTLY THE SAME ORDER as specified above
net_display_name_translation = {
  "inception"  : "Inceptionv3",
  "resnet"     : "ResNet152",
  "senet"      : "SENet154",
  "bert"       : "BERT",
  "vit"        : "ViT",
  "resnext"    : "ResNeXt101_32",
}

net_name_translation = {
  "inception"  : "Inceptionv3",
  "resnet"     : "ResNet152",
  "senet"      : "SENet154",
  "bert"       : "BERT_Base",
  "vit"        : "VIT",
  "resnext"    : "ResNeXt101_32",
}

net_name_detail_translation = {
  "inception"  : "Inceptionv3|B = 1536|M = 1969.46%",
  "resnet"     : "ResNet152|B = 1280|M = 2715.45%",
  "senet"      : "SENet154|B = 1024|M = 4277.81%",
  "bert"       : "BERT|B = 256|M = 370.10%",
  "vit"        : "ViT|B = 1280|M = 461.11%"
}

# Memory Overcommitment
# SENet154    256  463.902851 GB/40.000000 GB (1159.757128%)
#             512  876.536873 GB/40.000000 GB (2191.342182%)
#             768  1289.170570 GB/40.000000 GB (3222.926426%)
#             1024 1711.122635 GB/40.000000 GB (4277.806587%)
# BERT        128  74.426689 GB/40.000000 GB (186.066723%)
#             256  148.041496 GB/40.000000 GB (370.103741%)
#             384  221.656303 GB/40.000000 GB (554.140759%)
#             512  295.271111 GB/40.000000 GB (738.177776%)
#             640  368.885918 GB/40.000000 GB (922.214794%)
#             768  442.500725 GB/40.000000 GB (1106.251812%)
#             1024 589.730339 GB/40.000000 GB (1474.325848%)
# ViT         256  37.071716 GB/40.000000 GB (92.679291%)
#             512  73.913918 GB/40.000000 GB (184.784794%)
#             768  110.757172 GB/40.000000 GB (276.892929%)
#             1024 147.599373 GB/40.000000 GB (368.998432%)
#             1280 184.442627 GB/40.000000 GB (461.106567%)
#             1536 221.284828 GB/40.000000 GB (553.212070%)
# Inceptionv3 512  264.217983 GB/40.000000 GB (660.544958%)
#             768  395.109856 GB/40.000000 GB (987.774639%)
#             1024 526.001720 GB/40.000000 GB (1315.004301%)
#             1152 591.447659 GB/40.000000 GB (1478.619146%)
#             1280 656.893593 GB/40.000000 GB (1642.233982%)
#             1408 722.339531 GB/40.000000 GB (1805.848827%)
#             1536 787.785465 GB/40.000000 GB (1969.463663%)
# ResNet152   256  229.904152 GB/40.000000 GB (574.760380%)
#             512  440.597485 GB/40.000000 GB (1101.493711%)
#             768  655.791191 GB/40.000000 GB (1639.477978%)
#             1024 870.984901 GB/40.000000 GB (2177.462254%)
#             1280 1086.178421 GB/40.000000 GB (2715.446053%)
#             1536 1301.371941 GB/40.000000 GB (3253.429852%)
