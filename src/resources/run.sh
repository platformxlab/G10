#!/bin/bash

print_help() {
  printf "Useage: %s:\n" "$1" >&2
  printf "Run simulation\n" >&2
  printf "  -h             print help, this message\n" >&2
  printf "  -g             regenerate all configs using genconfigs.py\n" >&2
  printf "  -d             remove all invalid output files\n" >&2
  printf "  -r             run all configs that does not have output\n" >&2
  printf "  -f             run all configs that have valid output to regenerate final stat\n" >&2
  printf "  -k             remove all output, comfirmation required\n" >&2
  printf "  -p [REGEX]     match specific config pattern\n" >&2
  printf "  -j [NUM_PROC]  max number of concurrent simulations\n" >&2
  printf "  -dr            rerun all configs that either invalid or not generated\n" >&2
  printf "  -y             ignore all confirmation, assert Y everytime (NOT recommanded)\n" >&2
  printf "Return values\n" >&2
  printf "  0              script terminates correctly\n" >&2
  printf "  1              invalid options\n" >&2
  printf "  2              abort on removal of critical files\n" >&2
  printf "  3              abort on simulation launching\n" >&2
  printf "  4              abort on required resources invalid/missing\n" >&2
}

# TODO:  integrate color removal to output log
# sed -i -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2};?)?)?[mGK]//g" [filename]

# abs path to src folder
src_folder=$(dirname "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)")
# abs path to results folder
results_folder=$(dirname "$src_folder")/results
# abs path to configs folder
configs_folder="$src_folder/configs"

# options to be set
RERUN_FAILED=0
RUN_VALID=0
REMOVE_INVALID=0
RM_EVERYTHING=0
GEN_CONFIG=0
OPTION_OVERRIDE=0

APPLICATION=gpg
GEN_SCRIPT=resources/genconfigs.py
MAX_CONCURRENT_RUN=25
PATTERN=".*"

display_yesno_option() {
  printf "(Y/[n]) ? " >&2
  if [ $OPTION_OVERRIDE -ne 0 ]; then 
    printf "Y\n" >&2
    selection="Y"
  else
    read -r selection
  fi
  return $(test "$selection" == "Y")
}

# parse command line args
while getopts "hgdrfkyp:j:" arg; do
  case $arg in
    h)
      print_help "$@"
      exit 0
    ;;
    g)
      GEN_CONFIG=1
    ;;
    d)
      process_cnt=$(pgrep $APPLICATION -u "$USER" | wc -l)
      if [[ $process_cnt -gt 0 ]]; then
        printf "\033[0;31mThere are %d gpg running, force clean\033[0m " "$process_cnt" >&2
        display_yesno_option
        if [ $? -eq 0 ]; then
          REMOVE_INVALID=1
        else
          printf "Abort\n" >&2
          exit 3
        fi
      else
        REMOVE_INVALID=1
      fi
    ;;
    r)
      RERUN_FAILED=1
    ;;
    f)
      RUN_VALID=1
    ;;
    k)
      RM_EVERYTHING=1
    ;;
    j)
      MAX_CONCURRENT_RUN=${OPTARG}
      re='^[0-9]+$'
      if ! [[ $MAX_CONCURRENT_RUN =~ $re ]]; then
        printf "MAX_CONCURRENT_RUN specified <%s> is not a number\n" "$MAX_CONCURRENT_RUN" >&2
        exit 1
      fi
    ;;
    p)
      PATTERN=${OPTARG}
      printf "PATTERN: %s\n" "$PATTERN"
    ;;
    y)
      OPTION_OVERRIDE=1
      printf "Overriding all options\n" "$MAX_CONCURRENT_RUN" >&2
    ;;
    *)
      print_help "$@"
      exit 1
    ;;
  esac
done

if [ $GEN_CONFIG -ne 0 ]; then
  if ! [ -f "$src_folder/$GEN_SCRIPT" ]; then
    printf "\033[0;31mConfig generation script <%s> not found\033[0m\n" "$src_folder/$GEN_SCRIPT" >&2
    exit 4
  fi
  IFS=" " read -ra overwrite_folders <<< "$(find "$configs_folder" -type d -not -path "$configs_folder" -print0 | xargs --null)"
  if [ "${#overwrite_folders[@]}" -ne 0 ]; then
    printf "Overwritting configs\n" >&2
    printf "  %s\n" "${overwrite_folders[@]}"
    printf "Above folders will be removed, continue " >&2
    display_yesno_option
    if [ $? -ne 0 ]; then
      exit 2
    fi
    printf "%s " "${overwrite_folders[@]}" | xargs rm -r
    printf "All configs removed, generating configs\n" >&2
  fi
  python3 "$src_folder/$GEN_SCRIPT"
  if [ $? -ne 0 ]; then
    printf "\033[0;33mConfig generation abort\033[0m\n" >&2
    exit 4
  fi
  printf "Config generation done\n" >&2
  exit 0
fi

# some variables
sim_configs=()
run_configs=()
rerun_configs=()
output_postfixs=( evc kernel pcie pf_tensor transfer_boundary )
final_output_postfix=final

# files=()
# readarray -d '' files < <(find "$configs_folder" -name "*.config" -print0)

# for file in "${files[@]}"; do
for file in "$src_folder"/configs/*/*.config; do
  config_name=$(basename "$file")
  config_folder=$(basename "$(dirname "$file")")
  config_display_name="$config_folder/$config_name"
  if ! [[ $config_display_name =~ $PATTERN ]]; then continue; fi
  ret=$(sed -rn "s/is_simulation\s*([0-9]+)\s*/\1/p" "$file" 2> /dev/null)
  if [ -n "$ret" ] && [ "$ret" -ge 1 ]; then
    sim_configs+=("$file")
  fi
done

if [ $RM_EVERYTHING -eq 1 ]; then
  printf "Will operate on\n" >&2
  for config in "${sim_configs[@]}"; do
    config_name=$(basename "$config")
    config_folder=$(basename "$(dirname "$config")")
    printf "%s %s\n" "$config_folder" "$config_name"
  done  
  printf "\033[0;31mREMOVE ALL OUTPUT\033[0m " >&2
  display_yesno_option
  if [ $? -ne 0 ]; then
    printf "Abort\n" >&2
    exit 2
  fi
fi

target_output_folders=()
printf "Total %d sim configs found\n" "${#sim_configs[@]}" >&2
invalid_config_cnt=0
missing_config_cnt=0
valid_config_cnt=0
for file in "${sim_configs[@]}"; do
  # retrieve output folder name from config file
  output_folder="$results_folder"/$(sed -rn "s/output_folder\s*(\S+)\s*/\1/p" "$file")
  if [[ $output_folder != */ ]]; then output_folder="$output_folder/"; fi
  # check for duplication
  if printf "%s\n" "${target_output_folders[@]}" | grep -Fxq -- "$output_folder"; then
    printf "Output folder <%s> duplicates, found in config <%s>. Abort\n" "$output_folder" "$file"
    exit 4
  fi
  target_output_folders+=("$output_folder")
  target_output_folders=($(sort -u <<< "${target_output_folders[@]}"))
  output_file_prefix=$(sed -rn "s/stat_output_file\s*(\S+)\s*/\1/p" "$file")
  config_name=$(basename "$file")
  config_folder=$(basename "$(dirname "$file")")
  config_display_name="$config_folder/$config_name"
  final_file="$output_folder$output_file_prefix.$final_output_postfix"
  if [ -d "$output_folder" ]; then
    if [ "$RM_EVERYTHING" -ne "0" ]; then
      printf " \033[0;31m-\033[0m <%s> \033[0;31mRM\033[0m\n" "$config_display_name" >&2
      rm -r "$output_folder"
    else
      invalid_cnt=0
      not_exist_cnt=0
      for output_postfix in "${output_postfixs[@]}"; do
        output_file="$output_folder$output_file_prefix.$output_postfix"
        if [ -f "$output_file" ]; then
          lastline=$(tail -n 1 "$output_file" | xargs echo -n)
          if [ "$lastline" != "-1" ]; then
            (( invalid_cnt = invalid_cnt + 1 ))
          fi
        else
          (( not_exist_cnt = not_exist_cnt + 1 ))
        fi
      done
      if [[ $(( invalid_cnt + not_exist_cnt )) == 0 ]]; then
        (( valid_config_cnt = valid_config_cnt + 1 ))
        printf " \033[0;32mo\033[0m <%s> " "$config_display_name" >&2
        if [[ $RUN_VALID -ne 0 ]]; then
          printf "\033[0;33mTOTAL STAT RERUN\033[0m " >&2
          rm -f "$final_file" 
          printf "\033[0;32mRUN\033[0m" >&2
        fi
        run_configs+=("$file")
        printf "\n" >&2
      else
        (( invalid_config_cnt = invalid_config_cnt + 1 ))
        printf " \033[0;33m-\033[0m <%s>: %d missing %d invalid " "$config_display_name" "$not_exist_cnt" "$invalid_cnt" >&2
        if [[ $REMOVE_INVALID -ne 0 ]]; then
          rm -r "$output_folder"
          printf "\033[0;31mOUT FOLDER REMOVED\033[0m " >&2
          if [[ $RERUN_FAILED -ne 0 ]]; then 
            printf "\033[0;32mRERUN\033[0m" >&2
            rerun_configs+=("$file")
          fi
        fi
        printf "\n" >&2
      fi 
    fi
  else
    (( missing_config_cnt = missing_config_cnt + 1 ))
    printf " \033[0;31mx\033[0m <%s>: output folder does not exist " "$config_display_name" >&2
    if [[ $RERUN_FAILED -ne 0 ]]; then printf "\033[0;32mRUN\033[0m" >&2; fi
    printf "\n" >&2
    rerun_configs+=("$file")
  fi
done

printf "Found \033[0;33m%d\033[0m invalid result(s), \033[0;31m%d\033[0m missing result(s) \
and \033[0;32m%d\033[0m valid result(s). Total %d\n" \
"$invalid_config_cnt" "$missing_config_cnt" "$valid_config_cnt" "${#sim_configs[@]}" >&2

if [[ $RERUN_FAILED -ne 0 ]] || [[ $RUN_VALID -ne 0 ]]; then
  if ! make -C "$src_folder"; then
    printf "\033[0;31mSource code compilation error, abort\033[0m\n" >&2
    exit 4
  fi
fi

if [[ $RERUN_FAILED -ne 0 ]]; then
  num_runs=0
  sessions=()
  printf "Rerun %d sim configs" "${#rerun_configs[@]}" >&2
  for file in "${rerun_configs[@]}"; do
    # maintain maximum running 
    if [ "$num_runs" -ge "$MAX_CONCURRENT_RUN" ]; then
      while [ "$num_runs" -ge "$MAX_CONCURRENT_RUN" ]; do
        sleep 10
        num_runs=$(pgrep $APPLICATION -u "$USER" | wc -l)
        printf "." >&2
      done
      printf "\n" >&2
    fi
    group_name=$(basename "$(dirname "$file")")
    config_name=$(sed s/\.config// <<< "$(basename "$file")")
    new_session=0
    if ! tmux has-session -t "$group_name" &> /dev/null; then
      tmux new-session -d -s "$group_name" -c "$src_folder"
      printf "session %s spawned\n" "$group_name" >&2
      sessions+=("$group_name")
      new_session=1
    fi
    if ! tmux has-session -t "$group_name:$config_name" &> /dev/null; then
      output_folder="$results_folder"/$(sed -rn "s/output_folder\s*(\S+)\s*/\1/p" "$file")
      log_file="$output_folder/run.log"
      mkdir -p "$output_folder"
      tmux new-window -t "$group_name" -n "$config_name" -c "$src_folder"
      tmux send-keys -t "$group_name:$config_name" "set -o pipefail" Enter # for auto exit to run successfully
      tmux send-keys -t "$group_name:$config_name" "stdbuf --output=L ./$APPLICATION $file | tee $log_file && exit" Enter
      (( num_runs = num_runs + 1 ))
    else
      printf "session %s:%s already exists\n" "$group_name" "$config_name" >&2
    fi
    if [ "$new_session" -ne 0 ]; then 
      tmux kill-window -t "$session:$(tmux display-message -p "#{base-index}")"
    fi
  done
  printf "All sim configs launched, waiting for completion" >&2
  while [ "$num_runs" -gt 0 ]; do
    sleep 10
    num_runs=$(pgrep $APPLICATION -u "$USER" | wc -l)
    printf "." >&2
  done
  printf "\nDone\n" >&2
fi

if [[ $RUN_VALID -ne 0 ]]; then
  num_runs=0
  sessions=()
  printf "Run %d sim configs" "${#run_configs[@]}" >&2
  for file in "${run_configs[@]}"; do
    # maintain maximum running 
    if [ "$num_runs" -ge "$MAX_CONCURRENT_RUN" ]; then
      while [ "$num_runs" -ge "$MAX_CONCURRENT_RUN" ]; do
        sleep 10
        num_runs=$(pgrep $APPLICATION -u "$USER" | wc -l)
        printf "." >&2
      done
      printf "\n" >&2
    fi
    group_name=$(basename "$(dirname "$file")")
    config_name=$(sed s/\.config// <<< "$(basename "$file")")
    new_session=0
    if ! tmux has-session -t "$group_name" &> /dev/null; then
      tmux new-session -d -s "$group_name" -c "$src_folder"
      printf "session %s spawned\n" "$group_name" >&2
      sessions+=("$group_name")
      new_session=1
    fi
    if ! tmux has-session -t "$group_name:$config_name" &> /dev/null; then
      output_folder="$results_folder"/$(sed -rn "s/output_folder\s*(\S+)\s*/\1/p" "$file")
      log_file="$output_folder/run_analysis.log"
      tmux new-window -t "$group_name" -n "$config_name" -c "$src_folder"
      tmux send-keys -t "$group_name:$config_name" "set -o pipefail" Enter # for auto exit to run successfully
      tmux send-keys -t "$group_name:$config_name" "stdbuf --output=L ./$APPLICATION $file | tee $log_file && exit" Enter
      (( num_runs = num_runs + 1 ))
    else
      printf "session %s:%s already exists\n" "$group_name" "$config_name" >&2
    fi
    if [ "$new_session" -ne 0 ]; then 
      tmux kill-window -t "$session:$(tmux display-message -p "#{base-index}")"
    fi
  done
  printf "All rerun configs launched, waiting for completion" >&2
  while [ "$num_runs" -gt 0 ]; do
    sleep 10
    num_runs=$(pgrep $APPLICATION -u "$USER" | wc -l)
    printf "." >&2
  done
  printf "\nDone\n" >&2
fi
