#!/bin/bash

# print help for this script
print_help() {
  echo "Useage: $1:" >&2
  echo "Compile and run the profiling -- Whole Mode" >&2
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
STATISTICS_FILE_DIR="$OUTPUT_DIR/statistics"

# compile all kernels
if [[ "$COMPILE" == 1 ]]; then\
  # compile in parallel
  echo "Compilation start" >&2
  make -C "$OUTPUT_DIR" -j "$MAX_THREAD_NUM" > /dev/null
  echo "Compilation done" >&2
else
  echo "Skip compilation" >&2
fi

# run and profile all kernels
if [[ "$RUN" == 1 ]]; then
  # sanity check
  if ! [[ -f $OUTPUT_DIR/main ]]; then
    echo "Compiled file does not exist, try compiling first" >&2
    exit 4
  fi
  # setup output file
  statistics_file="$STATISTICS_FILE_DIR/kernel_times.txt"
  cat /dev/null > "$statistics_file"
  # get number of kernels using pattern "// TOTAL_KERNEL_NUM: xxxx" that is embedded in generated code
  total_kernel_num=$(sed -n 's/\/\/ TOTAL_KERNEL_NUM: \([0-9]\+\).*/\1/p' "$OUTPUT_DIR/main.cu")

  echo "Profiling start on $total_kernel_num kernels" >&2
  # set output buffer to be buffered on line basis to allow pretty printing and failure resistence
  stdbuf -oL "$OUTPUT_DIR/main" | tee "$statistics_file" | python3 -m tqdm --total "$total_kernel_num" > /dev/null
  echo "Profiling done" >&2
else
  echo "Skip profiling" >&2
fi
