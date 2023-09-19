# Scripts for G10 Artifact

This directory contains a series of scripts for the user to conveniently run the experiments and generate results. 

### genconfigs.py
This Python script is used to automatically generate all the config files for the artifact experiments. The user can also modify it to generate customized config files. After running this script, the config files will be generated under the `src/configs` directory. 

Note: The baseline types supported are ["lru", "prefetch_lru", "deepUM", "FlashNeuron", "G10GDSSSD", "G10GDSFULL"]. Among these, the "lru"  represents the "Base_UVM" baseline; "prefetch_lru" represents "G10"; "G10GDSSSD" and "G10GDSFULL" represents "G10-GDS" and "G10-Host" baselines described in the paper, respectively. 


### run.sh
This shell script is used to conveniently run multiple experiments concurrently. It supports using regular expressions to match multiple config files, and it will automatically spawn different experiments to multiple `tmux` windows for parallel execution.

Usage: 
```bash
Run simulation
  -h             print help, this message
  -g             regenerate all configs using genconfigs.py
  -d             remove all invalid output files
  -r             run all configs that does not have output
  -f             run all configs that have valid output to regenerate final stat
  -k             remove all output, comfirmation required
  -p [REGEX]     match specific config pattern
  -j [NUM_PROC]  max number of concurrent simulations
  -dr            rerun all configs that either invalid or not generated
  -y             ignore all confirmation, assert Y everytime (NOT recommanded)
Return values
  0              script terminates correctly
  1              invalid options
  2              abort on removal of critical files
  3              abort on simulation launching
  4              abort on required resources invalid/missing
```

Examples of using this script can be found in `src/artifact_run.sh`. When you batch a number of experiments, you can use `tmux ls` to see all the spawned tmux sessions. You can also use `htop` to see the processes' information.


### gatherKernelInfo.py
This Python script is used to collect all the performance data already generated in the `results` directory. The raw data collected will be stored in `raw_output/data.json`.

### figureDrawingDataPrep*.py
These Python scripts are used to gather specific data for each figure by analyzing the `raw_output/data.json`. The data gathered for each figure is stored in the `figure_drawing/` directory. The figure drawing scripts are also placed in that directory.
