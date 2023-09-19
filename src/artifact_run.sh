# --------------------------------- IMPORTART -------------------------------------------------
# ! Please modify this number based on your machine's main memory capacity. One experiment process will need a peak memory of 28.5 GB.
# We recommend reserving 30 GB for each process to ensure that the program won't crash.
# For example, if your machine has 128 GB main mamory, this number can be set as 4.
MAX_PROCESS_NUM=4
# ---------------------------------------------------------------------------------------------

# Prerequisite: flex bison tmux
## sudo apt install flex bison tmux python3-pip
# Python prerequisite: matplotlib networkx pandas PyPDF2

make clean
make -j8
cd resources

# Generate config files
python3 genconfigs.py

#-------------------------------- Running Experiments -----------------------------------------------------------------------------------
# Make sure there is no G10 experiment running in the background (in tmux)
tmux kill-server

# First run experiments for figure 11-14
./run.sh -p "(BERT\/256|VIT\/1280|Inceptionv3\/1536|ResNet152\/1280|SENet154\/1024)-sim_(deepUM|prefetch_lru|FlashNeuron|G10GDSSSD|G10GDSFULL|lru)\.config" -dr -j $MAX_PROCESS_NUM
# The time for running this is about 104m33.975s (for MAX_PROCESS_NUM=6)

# Then run experiments for figure 15
./run.sh -p "(BERT\/(128|256|512|768|1024)|VIT\/(256|512|768|1024|1280)|Inceptionv3\/(512|768|1024|1280|1536|1792)|ResNet152\/(256|512|768|1024|1280)|SENet154\/(256|512|768|1024))-sim_(deepUM|prefetch_lru|FlashNeuron|lru)\.config" -dr -j $MAX_PROCESS_NUM
# The time for running this is about 155m11.104s (for MAX_PROCESS_NUM=6)

# Then run experiments for figure 16
./run.sh -p "(BERT\/(256|384|512|640)|VIT\/(768|1024|1280|1536)|Inceptionv3\/(512|1024|1280|1536)|ResNet152\/(768|1024|1280|1536)|SENet154\/(256|512|768|1024))-sim_prefetch_lru(-cpu(0|16|32|64|96|192|256))?\.config" -dr -j $MAX_PROCESS_NUM
# The time for running this is about 406m30.954s (for MAX_PROCESS_NUM=6)

# Then run experiments for figure 17
./run.sh -p "(VIT\/1024|Inceptionv3\/1280)-sim_(deepUM|prefetch_lru|FlashNeuron)-cpu(0|16|32|64|256)\.config" -dr -j $MAX_PROCESS_NUM
# The time for running this is about 24m8.144s (for MAX_PROCESS_NUM=6)

# Then run experiments for figure 18
./run.sh -p "(BERT\/512|VIT\/1280|Inceptionv3\/1536|ResNet152\/1280|SENet154\/1024)-sim_(deepUM|prefetch_lru|FlashNeuron|lru)-ssd(6_4|12_8|19_2|25_6|32)-.*\.config" -dr -j $MAX_PROCESS_NUM
# The time for running this is about 354m40.747s (for MAX_PROCESS_NUM=6)

# Then run experiments for figure 19
./run.sh -p "(BERT\/256|VIT\/1280|Inceptionv3\/1536|ResNet152\/1280|SENet154\/1024)-sim_prefetch_lru-var0_(05|10|15|20|25)\.config" -dr -j $MAX_PROCESS_NUM
# The time for running this is about 124m17.909s (for MAX_PROCESS_NUM=6)



#-------------------------------- Gathering Data -----------------------------------------------------------------------------------=

# Collect all the numbers, store it in raw_output/data.json
python3 gatherKernelInfo.py

# Gather data for figure 11
python3 figureDrawingDataPrepOverallPerformance.py  # The gathered data is stored in figure_drawing/overall_performance

# Gather data for figure 12
python3 figureDrawingDataPrepBreakdown.py  # The gathered data is stored in figure_drawing/overall_breakdown

# Gather data for figure 13
./figureDrawingDataPrepKernelCDF.sh  # The gathered data is stored in figure_drawing/overall_slowdown_cdf

# Gather data for figure 14
python3 figureDrawingDataPrepTraffic.py  # The gathered data is stored in figure_drawing/overall_traffic

# Gather data for figure 15
python3 figureDrawingDataPrep.py  # The gathered data is stored in figure_drawing/overall_batchsize

# Gather data for figure 16
python3 figureDrawingDataPrepCPUsensitivity.py  # The gathered data is stored in figure_drawing/sensitivity_cpumem

# Gather data for figure 17
python3 figureDrawingDataPrepCPUSensitivityCombined.py  # The gathered data is stored in figure_drawing/sensitivity_cpumem_combined

# Gather data for figure 18
python3 figureDrawingDataPrepSSD.py  # The gathered data is stored in figure_drawing/sensitivity_ssdbw

# Gather data for figure 19
python3 figureDrawingDataPrepVariation.py  # The gathered data is stored in figure_drawing/sensitivity_variation



#-------------------------------- Drawing Figures -----------------------------------------------------------------------------------

cd figure_drawing

# Plot figures for Figure 2-4, and Figure 20-21 (Appendix)

python3 plot_mem_consumption.py  # Figure 2 is output/dnn_memconsumption.pdf

python3 plot_tensor_time_cdf.py  # Figure 3 is output/tensor_time_cdf.pdf

python3 plot_tensor_period_distribution.py  # Figure 4 is output/tensor_periods_distribution.pdf

python3 plot_detail_mem_breakdown_live.py  # Figure 20 is output/dnn_mem_consumption_breakdown_live.pdf

python3 plot_detail_mem_breakdown_active.py  # Figure 21 is output/dnn_mem_consumption_breakdown_active.pdf

# Draw Figure 11
python3 overallPerf.py  # Figure 11 is output/OverallPerfNew.pdf

# Draw Figure 12
python3 overallBreakdown.py  # Figure 12 is output/Breakdown.pdf

# Draw Figure 13
python3 overallSlowdownCDF.py  # Figure 13 is output/KernelTimeCDF.pdf

# Draw Figure 14
python3 overallTraffic.py  # Figure 14 is output/OverallTraffic.pdf

# Draw Figure 15
python3 overallBatchSize.py  # Figure 15 is output/OverallPerfBatchSize.pdf

# Draw Figure 16
python3 sensitivityCPUMem.py  # Figure 16 is output/OverallPerfCPUMem.pdf

# Draw Figure 17
python3 sensitivityCPUMemCombined.py  # Figure 17 is output/OverallPerfCPUMemCombined.pdf

# Draw Figure 18 
python3 sensitivitySSDbw.py  # Figure 18 is output/OverallPerfSSDBW.pdf 

# Draw Figure 19
python3 SensitivityKernelVariation.py # Figure 19 is output/SensitivityVariation.pdf
