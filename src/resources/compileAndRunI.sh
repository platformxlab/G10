#!/bin/bash

# print help for this script
print_help() {
  echo "Useage: $1:" >&2
  echo "Compile and run the profiling -- Individual Mode" >&2
  echo "  -h             print help, this message" >&2
  echo "  -c             auto compile all the profiling applications" >&2
  echo "  -r             run all the compiled profiling applications" >&2
  echo "  -t [NUM_CORES] specify the number of cores to run this application" >&2
  echo "                 default to be the number of logical processors get" >&2
  echo "                 from using <nproc --all> command" >&2
}

# options to be set
MAX_THREAD_NUM=$(nproc --all)
COMPILE=0
RUN=0
# parse command line args
while getopts "hcrt:" arg; do
  case $arg in
    h)
      print_help "$@"
      exit 0
    ;;
    c)
      COMPILE=1
    ;;
    r)
      RUN=1
    ;;
    t)
      MAX_THREAD_NUM=${OPTARG}
      re='^[0-9]+$'
      if ! [[ $MAX_THREAD_NUM =~ $re ]]; then
        echo "MAX_THREAD_NUM specified <$MAX_THREAD_NUM> is not a number" >&2
        exit 1
      fi
    ;;
    *)
      print_help "$@"
      exit 1
    ;;
  esac
done
# do nothing
if [[ $COMPILE == 0 && $RUN == 0 ]]; then
  echo "Both Compile and Run are disabled, do nothing" >&2
  exit 0
fi

echo

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit 255

OUTPUT_DIR=$(dirname "$SCRIPT_DIR")
PROFILING_FILE_DIR="$OUTPUT_DIR/profiling_src"
OTHER_SOURCE_FILE_DIR="$OUTPUT_DIR/src"
BINARY_FILE_DIR="$OUTPUT_DIR/bin"
STATISTICS_FILE_DIR="$OUTPUT_DIR/statistics"
# only files that are in the format of profiling*.cu are considered as kernel
source_files=( "$PROFILING_FILE_DIR/profiling*.cu" )
total_kernel_num=${#source_files[@]}

# compile all kernels
if [[ "$COMPILE" == 1 ]]; then
  # compile in parallel
  echo "Compilation start on $total_kernel_num kernels" >&2
  echo "Spawning processes at maximum $MAX_THREAD_NUM" >&2
  iter_num=$(( (${#source_files[@]} + $MAX_THREAD_NUM - 1) / $MAX_THREAD_NUM ))
  # spawning thread at maximum $MAX_THREAD_NUM threads each iteration of the loop
  for file_index_base in $(seq 0 $MAX_THREAD_NUM $(( ${#source_files[@]} - 1))); do
    remaining_len=$(( ${#source_files[@]} - $file_index_base ))
    if (( $remaining_len > $MAX_THREAD_NUM )); then
      remaining_len=$MAX_THREAD_NUM
    fi
    # creating individual threads
    for file_index in $(seq "$file_index_base" $(( $file_index_base + $remaining_len - 1))); do
      source_file=${source_files[$file_index]}
      source_name=$(basename -- "$source_file")
      bin_name="$BINARY_FILE_DIR/${source_name%.*}.bin"
      # Find other source files by parsing the first line of the *.cu file. Because the generated
      # files are always begin with one and only one inclusion that indiciates another declaration
      # file it would use, extract that file with regex expression and add that to nvcc compilier
      other_source_files=$(head -n 1 "$source_file")
      other_source_files=$(expr "$other_source_files" : '#include "\(.*\)"')
      other_source_files=$(basename -- "$other_source_files")
      other_source_files="$OTHER_SOURCE_FILE_DIR/${other_source_files%.*}.cu"
      # if a compiled binary is already there, skip it
      if [[ ! -f "$bin_name" ]]; then
        nvcc -o "$bin_name" "$source_file" "$other_source_files" &
      fi
    done
    wait # block until all spawned processes finished
    echo # for tqdm to work correctly
  done | python3 -m tqdm --total "$iter_num" > /dev/null
  echo "Compilation done" >&2
else
  echo "Skip compilation" >&2
fi

# run and profile all kernels
if [[ "$RUN" == 1 ]]; then
  # get all binary files
  bin_files=()
  for source_file in "${source_files[@]}"; do
    source_name=$(basename -- "$source_file")
    bin_name="$BINARY_FILE_DIR/${source_name%.*}.bin"
    bin_files+=( "$bin_name" )
  done

  # run binary in serial
  echo "Profiling start" >&2
  statistics_file="$STATISTICS_FILE_DIR/kernel_times.txt"
  cat /dev/null > "$statistics_file"
  for bin_name in "${bin_files[@]}"; do
    "$bin_name" >> "$statistics_file"
    echo # for tqdm to work correctly
  done | python3 -m tqdm --total "$total_kernel_num" > /dev/null
  echo "Profiling done" >&2
else
  echo "Skip profiling" >&2
fi
