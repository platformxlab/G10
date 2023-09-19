/* File: main.cc
 * -------------
 * This file defines the main() routine for the program and not much else.
 * You should not need to modify this file.
 */

#include <chrono>
#include <string>
#include <math.h>
#include <random>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <unistd.h>
#include "utility.h"
#include "errors.h"
#include "parser.h"
#include "y.tab.h"
#include "analysis.h"
#include "codegen.h"
#include "simulationComponents.h"
#include "simulator.h"
#include "printUtils.h"

#define YYDEBUG 1

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// Codegen param
extern int is_resnet;
extern int is_inception;
extern int is_senet;
extern int batch_size;
extern int input_H;
extern int input_W;
extern int num_threads;
extern bool is_individual;
extern bool is_input_pf_only;

// CPU sim param
extern double CPU_PCIe_bandwidth_GBps;
// GPU sim param
extern double GPU_PCIe_bandwidth_GBps;
extern double GPU_frequency_GHz;
extern double GPU_memory_size_GB;
extern double GPU_malloc_uspB;
extern double GPU_free_uspB; // NOT USED FOR NOW
// SSD sim param
extern double SSD_PCIe_bandwidth_GBps;
// PCIe sim param
extern double PCIe_latency_us;  // NOT USED FOR NOW
extern int PCIe_batch_size_in_page;
// Other sim param
extern bool use_prefetch;
extern std::string migration_policy_str;
extern std::string eviction_policy_str;
extern Simulator::MigPolicy migration_policy;
extern Simulator::GPUPageTable::EvcPolicy eviction_policy;
extern int prefetch_degree;
extern int num_candidate;
extern double system_latency_us; // NOT USED FOR NOW

// Other param
//   In codegen, is_UVM specifies whether to use cudaMallocManaged
//   In simulation, is_UVM specifies whether setup is ideal (i.e. all tensor in GPU mem)
bool is_UVM = true;
//   In codegen, num_iteration specifies number of iterations to profile
//   In simulation, num_iteration specifies number of iterations to run
int num_iteration = -1;
int is_transformer = -1;
int borden = 184;

// 
extern double CPU_memory_line_GB;
extern double SSD_read_latency_us;
extern double SSD_write_latency_us;
extern double SSD_latency_us; // Upper bound
extern double delta_parameter;

// Tensor configurations
extern long long memory_offset_intermediate;
extern long long memory_offset_weights;

// 
extern std::vector<Model_Layer*> forward_layers;
extern std::vector<Model_OP*> forward_ops;
extern std::vector<CUDAKernel> kernel_list;
extern std::vector<Tensor*> tensor_list;
extern std::vector<Hidding_Interval*> interval_list;
extern std::vector<EvictionGuide_Entry> EvictionGuide_Table;
extern std::vector<long> GPU_resident_memory_estimation;
extern std::vector<double> kernel_time_table;

// output specifications
std::string nn_model_input_file;
std::string orig_kernel_time_file;
std::string input_pf_kernel_time_file;
std::string workspace_size_file;
std::string pf_kernel_time_file;
std::string stat_output_file;
std::string output_folder_name;
// simulation switches
bool is_simulation = true;
bool output_override = false;
// profiling switches
bool is_compile = true;
bool is_run = true;
int compile_max_thread_num = -1;
bool is_cudnn = false;

// random devices
std::mt19937 rand_device;
double kernel_time_std_dev = 0;
unsigned int ran_seed = 1;
double kernel_speedup = 1;

/* Function: PrintOneToken()
 * Usage: PrintOneToken(T_Double, "3.5", val, loc);
 * -----------------------------------------------
 */
static void PrintOneToken(yytokentype token, const char *text, YYSTYPE value,
                          yyltype loc)
{
  char buffer[] = {'\'', (char) token, '\'', '\0'};
  const char *name = token >= T_Sequential ? gTokenNames[token - T_Sequential] : buffer;

  printf("%-12s line %d cols %d-%d is %s ", text,
	   loc.first_line, loc.first_column, loc.last_column, name);

  switch(token) {
    case T_IntConstant:
      printf("(value = %d)\n", value.integerConstant); break;
    case T_DoubleConstant:
      printf("(value = %g)\n", value.doubleConstant); break;
    case T_BoolConstant:
      printf("(value = %s)\n", value.boolConstant ? "true" : "false"); break;
    case T_Identifier:
	if (strcmp(text, value.identifier)) {
	  printf("(truncated to %s)\n", value.identifier);
	  break;
	}
    default:
      printf("\n"); break;
  }
}

void CheckVar(double var, std::string variable_name, bool gt=true) {
    if ((gt && var < 0) || (!gt && var > 0)) {
        eprintf("Invalid or missing <%s>, current value: %f, should be %s than 0, aborting\n", 
                variable_name.c_str(), var, gt ? "greater" : "less");
        Assert(false);
    }
}

void SimulationParamSanityCheck() {
    // parameter validation (existence)
    CheckVar(PCIe_batch_size_in_page, "PCIe_batch_size_in_page");
    CheckVar(CPU_PCIe_bandwidth_GBps, "CPU_PCIe_bandwidth_GBps");
    CheckVar(GPU_PCIe_bandwidth_GBps, "GPU_PCIe_bandwidth_GBps");
    CheckVar(SSD_PCIe_bandwidth_GBps, "SSD_PCIe_bandwidth_GBps");
    CheckVar(GPU_frequency_GHz, "GPU_frequency_GHz");
    CheckVar(GPU_memory_size_GB, "GPU_memory_size_GB");
    CheckVar(GPU_malloc_uspB, "GPU_malloc_uspB");
    CheckVar(GPU_free_uspB, "GPU_free_uspB");
    CheckVar(CPU_memory_line_GB, "CPU_memory_line_GB");

    if (migration_policy == Simulator::MigPolicy::DEEPUM)
        Assert(eviction_policy == Simulator::GPUPageTable::EvcPolicy::DEEPUM);
    if (eviction_policy == Simulator::GPUPageTable::EvcPolicy::DEEPUM)
        Assert(migration_policy == Simulator::MigPolicy::DEEPUM);
    if (migration_policy == Simulator::MigPolicy::DEEPUM)
        CheckVar(prefetch_degree, "prefetch_degree");
    else
        CheckVar(prefetch_degree, "prefetch_degree", false);
    if (eviction_policy == Simulator::GPUPageTable::EvcPolicy::GUIDED || 
        eviction_policy == Simulator::GPUPageTable::EvcPolicy::GUIDED_LRU ||
        eviction_policy == Simulator::GPUPageTable::EvcPolicy::PERFECT_GUIDED ||
        eviction_policy == Simulator::GPUPageTable::EvcPolicy::PERFECT_GUIDED_LRU)
        CheckVar(num_candidate, "num_candidate");
    else
        CheckVar(num_candidate, "num_candidate", false);
    CheckVar(num_iteration, "num_iteration");

    // parameter validation (value)
    if (SSD_PCIe_bandwidth_GBps > GPU_PCIe_bandwidth_GBps) {
        eprintf("Invalid SSD Bandwidth [%f] > GPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, GPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (CPU_PCIe_bandwidth_GBps > GPU_PCIe_bandwidth_GBps) {
        eprintf("Invalid CPU Bandwidth [%f] > GPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, GPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (SSD_PCIe_bandwidth_GBps > CPU_PCIe_bandwidth_GBps) {
        eprintf("Unsupported SSD Bandwidth [%f] > CPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, CPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (GPU_PCIe_bandwidth_GBps > SSD_PCIe_bandwidth_GBps + CPU_PCIe_bandwidth_GBps) {
        eprintf("Unsupported GPU Bandwidth [%f] > SSD Bandwidth [%f] + CPU Bandwidth [%f]\n",
                GPU_PCIe_bandwidth_GBps, SSD_PCIe_bandwidth_GBps, CPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (kernel_speedup <= 0) {
        eprintf("Invalid kernel speedup [%f]\n", kernel_speedup);
        Assert(false);
    }
}

void SetupOutputFolder() {
    if (output_override)
        wprintf("Overriding output folder <%s>...\n", output_folder_name.c_str());
    Assert(system(("mkdir -p " + output_folder_name).c_str()) == 0);
    Assert(system(("find " + output_folder_name + "/statistics -name \"*.config\" -type f | xargs rm -f").c_str()) == 0);
    // clean up dirs
    if (output_override && !is_simulation) {
        Assert(system(("rm -rf " + output_folder_name + "/include").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/src").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/bin").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/scripts").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/profiling_src").c_str()) == 0);
        Assert(system(("rm -f " + output_folder_name + "/main.cu").c_str()) == 0);
        Assert(system(("rm -f " + output_folder_name + "/main").c_str()) == 0);
    }
    // make dirs
    Assert(system(("mkdir -p " + output_folder_name + "/statistics").c_str()) == 0);
    // LRU visualization ////////////////////////////////////////////////////////////
    // Assert(system(("mkdir -p " + output_folder_name + "/lru_trace").c_str()) == 0);
    /////////////////////////////////////////////////////////////////////////////////
    if (!is_simulation) {
        Assert(system(("mkdir -p " + output_folder_name + "/include").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/src").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/bin").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/scripts").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/profiling_src").c_str()) == 0);
        Assert(system(("cp ./resources/cudadnnUtil.cuh " + output_folder_name + "/include/cudadnnUtil.cuh").c_str()) == 0);
        Assert(system(("cp ./resources/cudadnnUtil.cu " + output_folder_name + "/src/cudadnnUtil.cu").c_str()) == 0);
        Assert(system(("cp ./resources/Makefile " + output_folder_name + "/Makefile").c_str()) == 0);
        if (is_individual) {
            Assert(system(("cp ./resources/compileAndRunI.sh " + output_folder_name + "/scripts/compileAndRun.sh").c_str()) == 0);
        } else {
            Assert(system(("cp ./resources/compileAndRunW.sh " + output_folder_name + "/scripts/compileAndRun.sh").c_str()) == 0);
        }
    }
}

void loadWorkspaceSizes() {
    std::ifstream wok_f(workspace_size_file);
    Assert(wok_f.good());

    int kernel_id;
    string workspace_size_str;
    string unit;
    size_t workspace_size;
    iprintf("Loading workspace sizes from file <%s> for %d kernels\n",
            workspace_size_file.c_str(), kernel_list.size());
    for (int i = 0; i < kernel_list.size(); i++) {
        workspace_size_str = "";
        wok_f >> kernel_id >> workspace_size_str >> unit;
        Assert(kernel_id == i);
        Assert(workspace_size_str != "");
        Assert(unit == "B");
        workspace_size = std::stoull(workspace_size_str);
        Assert(workspace_size >= 0);
        if (workspace_size > 0) {
            kernel_list[i].workspace = new Tensor(workspace_size, false);
            kernel_list[i].outputs.insert(kernel_list[i].workspace);
            tensor_list.push_back(kernel_list[i].workspace);
        }
        
    }
    iprintf("Loading workspace sizes done\n\n", "");
}




void loadKernelTimes() {
    double GPU_frequency_Hz = GPU_frequency_GHz * pow(10, 9);
    
    std::ifstream orig_f(orig_kernel_time_file);
    std::ifstream pf_f(pf_kernel_time_file);
    std::ifstream inputpf_f(input_pf_kernel_time_file);
    Assert(orig_f.good());
    Assert(pf_f.good());
    Assert(inputpf_f.good());

    int kernel_num;
    string exe_time_ms_str; 
    string unit;
    // read in all the execution times
    long exe_time_cycle;
    double total_time = 0, pf_total_time = 0, input_pf_total_time = 0;
    unsigned long total_time_cycle = 0, pf_total_time_cycle = 0, input_pf_total_time_cycle = 0;
    iprintf("Loading kernel times from file <%s> and <%s> and <%s> for %d kernels\n",
            orig_kernel_time_file.c_str(), pf_kernel_time_file.c_str(), input_pf_kernel_time_file.c_str(), kernel_list.size());
    if (kernel_speedup != 1) {
        iprintf("Using kernel speedup of %.4fx\n", kernel_speedup);
    }
    for (int i = 0; i < kernel_list.size(); i++) {
        double delta_execution_time;
        // read in ideal execution time from file
        exe_time_ms_str.clear();
        orig_f >> kernel_num >> exe_time_ms_str >> unit;
        Assert(kernel_num == i);
        Assert(exe_time_ms_str != "");
        Assert(unit == "ms");
        exe_time_cycle = std::stod(exe_time_ms_str) * GPU_frequency_Hz / 1000.0;
        delta_execution_time = exe_time_cycle - exe_time_cycle / kernel_speedup;
        kernel_list[i].execution_cycles = exe_time_cycle - delta_execution_time;
        Assert(kernel_list[i].execution_cycles > 0);
        total_time += kernel_list[i].execution_cycles / GPU_frequency_Hz * 1000;
        total_time_cycle += exe_time_cycle;
        // read in input_pf execution time from file
        exe_time_ms_str.clear();
        inputpf_f >> kernel_num >> exe_time_ms_str >> unit;
        Assert(kernel_num == i);
        Assert(exe_time_ms_str != "");
        Assert(unit == "ms");
        exe_time_cycle = std::stod(exe_time_ms_str) * GPU_frequency_Hz / 1000.0;
        kernel_list[i].input_pf_execution_cycles = exe_time_cycle - delta_execution_time;
        if (kernel_list[i].input_pf_execution_cycles < kernel_list[i].execution_cycles)
            kernel_list[i].input_pf_execution_cycles = kernel_list[i].execution_cycles;
        // read in pf execution time from file
        exe_time_ms_str.clear();
        pf_f >> kernel_num >> exe_time_ms_str >> unit;
        Assert(kernel_num == i);
        Assert(exe_time_ms_str != "");
        Assert(unit == "ms");
        exe_time_cycle = std::stod(exe_time_ms_str) * GPU_frequency_Hz / 1000.0;
        kernel_list[i].pf_execution_cycles = exe_time_cycle - delta_execution_time;
        if (kernel_list[i].pf_execution_cycles < kernel_list[i].input_pf_execution_cycles)
            kernel_list[i].pf_execution_cycles = kernel_list[i].input_pf_execution_cycles;
        Assert(kernel_list[i].pf_execution_cycles > 0);
        Assert(exe_time_cycle > 0);
        pf_total_time += kernel_list[i].pf_execution_cycles / GPU_frequency_Hz * 1000;
        pf_total_time_cycle += exe_time_cycle;
        Assert(kernel_list[i].input_pf_execution_cycles > 0);
    }
    nprintf("Total time (Ideal): %f ms %lu cycles; (PF): %f ms %lu cycles\n", 
            total_time, total_time_cycle, pf_total_time, pf_total_time_cycle);
    // make sure kernel times file have no other entries left
    exe_time_ms_str = "";
    orig_f >> exe_time_ms_str;
    Assert(exe_time_ms_str == "");
    // make sure pf kernel times file have no other entries left
    exe_time_ms_str = "";
    pf_f >> exe_time_ms_str;
    Assert(exe_time_ms_str == "");
    // make sure inputpf kernel times file have no other entries left
    exe_time_ms_str = "";
    inputpf_f >> exe_time_ms_str;
    Assert(exe_time_ms_str == "");
    iprintf("Loading kernel times done\n\n", "");
}

class RedirStdOut {
    public:
        RedirStdOut(std::string filename) {
            info_file = output_folder_name + "/statistics/" + filename;
            buffer.str("");
            old_cout_buf = std::cout.rdbuf();
            cout_buf = std::cout.rdbuf(buffer.rdbuf());
            printf("Saving %s\n", filename.c_str());
        }
        ~RedirStdOut() {
            std::ofstream fout(info_file.c_str());
            fout << buffer.str();
            fout.close();
            std::cout.rdbuf(old_cout_buf);
        }
    private:
        std::string info_file;
        std::stringstream buffer;
        std::streambuf *old_cout_buf;
        std::streambuf *cout_buf;
};

/* Function: main()
 * ----------------
 * Entry point to the entire program.  We parse the command line and turn
 * on any debugging flags requested by the user when invoking the program.
 * InitScanner() is used to set up the scanner.
 * InitParser() is used to set up the parser. The call to yyparse() will
 * attempt to parse a complete program from the input.
 */
int main(int argc, char *argv[]) {
    // config file should be the first argument
    if (argc == 1) {
        eprintf("Please specify a config file\n", "");
        Assert(false);
    }
    // exit if config file does not exist
    std::ifstream config_file(argv[1]);
    if (!config_file.good()) {
        eprintf("Config file <%s> does not exist\n", argv[1]);
        Assert(false);
    }
    // parse config file
    std::string line; 
    std::string command;
    std::string value;
    printf("\nConfigs:\n");
    while (std::getline(config_file, line)) {
        std::stringstream ss(line);
        command.clear();
        value.clear();
        ss >> command >> value;
        if (command != "#" && command != "")
            printf("  %25s: <%s>\n", command.c_str(), value.c_str());

        // general settings
        if (command == "output_folder")                 { output_folder_name = value; }
        else if (command == "output_override")          { output_override = std::stoi(value) != 0; }
        else if (command == "is_simulation")            { is_simulation = std::stoi(value) != 0; }
        else if (command == "is_profiling")             { is_simulation = std::stoi(value) == 0; }
        // profiling general settings
        else if (command == "is_individual")            { is_individual = std::stoi(value) != 0; }
        else if (command == "is_compile")               { is_compile = std::stoi(value) != 0; }
        else if (command == "compile_max_thread_num")   { compile_max_thread_num = std::stoi(value); }
        else if (command == "is_run")                   { is_run = std::stoi(value) != 0; }
        else if (command == "is_cudnn")                 { is_cudnn = std::stoi(value) != 0; }
        // codegen settings
        else if (command == "is_resnet")                { is_resnet = std::stoul(value); }
        else if (command == "is_inception")             { is_inception = std::stoul(value);}
        else if (command == "is_senet")                 { is_senet = std::stoul(value);}
        else if (command == "is_transformer")           { is_transformer = std::stoi(value); }
        else if (command == "trans_borden")             { borden = std::stoi(value); }
        else if (command == "batch_size")               { batch_size = std::stoi(value); }
        else if (command == "input_H")                  { input_H = std::stoi(value); }
        else if (command == "input_W")                  { input_W = std::stoi(value); }
        else if (command == "num_iteration")            { num_iteration = std::stoi(value); }
        else if (command == "num_threads")              { num_threads = std::stoi(value); }
        else if (command == "is_input_pf_only")         { is_input_pf_only = std::stoi(value) != 0; }
        // simulation general settings
        else if (command == "is_UVM")                   { is_UVM = std::stoi(value) != 0; }
        else if (command == "use_prefetch")             { use_prefetch = std::stoi(value) != 0; }
        else if (command == "nn_model_input_file")      { nn_model_input_file = value; }
        else if (command == "orig_kernel_time_file")    { orig_kernel_time_file = value; }
        else if (command == "workspace_size_file")      { workspace_size_file = value; }
        else if (command == "input_pf_kernel_time_file"){ input_pf_kernel_time_file = value; }
        else if (command == "pf_kernel_time_file")      { pf_kernel_time_file = value; }
        else if (command == "stat_output_file")         { stat_output_file = value; }
        else if (command == "migration_policy")         { migration_policy_str = value; }
        else if (command == "eviction_policy")          { eviction_policy_str = value; }
        else if (command == "num_candidate")            { num_candidate = std::stoul(value); }
        else if (command == "prefetch_degree")          { prefetch_degree = std::stoi(value); }
        else if (command == "delta_parameter")          { delta_parameter = std::stod(value); }
        else if (command == "system_latency_us")        { system_latency_us = std::stod(value); }
        // simulation CPU statistics
        else if (command == "CPU_PCIe_bandwidth_GBps")  { CPU_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "CPU_memory_line_GB")       { CPU_memory_line_GB = std::stod(value); }
        // simulation GPU statistics
        else if (command == "GPU_PCIe_bandwidth_GBps")  { GPU_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "GPU_memory_size_GB")       { GPU_memory_size_GB = std::stod(value); }
        else if (command == "GPU_frequency_GHz")        { GPU_frequency_GHz = std::stod(value); }
        else if (command == "GPU_malloc_uspB")          { GPU_malloc_uspB = std::stod(value); }
        else if (command == "GPU_free_uspB")            { GPU_free_uspB = std::stod(value); }
        // simulation SSD statistics
        else if (command == "SSD_PCIe_bandwidth_GBps")  { SSD_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "SSD_read_latency_us")      { SSD_read_latency_us = std::stod(value); }
        else if (command == "SSD_write_latency_us")     { SSD_write_latency_us = std::stod(value); }
        else if (command == "SSD_latency_us")           { SSD_latency_us = std::stod(value); }
        // simulation PCIe statistics
        else if (command == "PCIe_latency_us")          { PCIe_latency_us = std::stod(value); }
        else if (command == "PCIe_batch_size_page")     { PCIe_batch_size_in_page = std::stoi(value); }
        // simulation Timing sentivity statistics
        else if (command == "kernel_time_std_dev")      { kernel_time_std_dev = std::stod(value); }
        else if (command == "ran_seed")                 { ran_seed = std::stoi(value); }
        else if (command == "kernel_speedup")           { kernel_speedup = std::stod(value); }
        // comments or empty line
        else if (command == "#" || command == "")       {}
        else {
          eprintf("Error: Invalid config entry <%s>, aborting...\n", command.c_str());
          Assert(false);
        }
    }
    // sanity check
    Assert((int) Simulator::GPUPageTable::EvcPolicy::DEEPUM != (int) Simulator::MigPolicy::DEEPUM);

    // indirection if there is no file is fed through stdin
    if (isatty(fileno(stdin))) {
      if (nn_model_input_file.empty()) {
        eprintf("No input NN model in either stdin or config file\n", "");
      } else {
        // open a file and redirect to stdin
        std::ifstream nn_model(nn_model_input_file.c_str());
        if (!nn_model.good()) {
            eprintf("Invalid input NN model specified in config file <%s>\n", 
                    nn_model_input_file.c_str());
            Assert(false);
        }
        if (is_transformer!=1)
        {
            freopen(nn_model_input_file.c_str(), "r", stdin);
        }
      }
    }
    // parameter transformation
    if (output_folder_name.back() == '/') output_folder_name.pop_back();
    stat_output_file = output_folder_name + "/" + stat_output_file;

    if (is_simulation) {
        // eviction policy
        std::transform(eviction_policy_str.begin(), eviction_policy_str.end(), eviction_policy_str.begin(), ::toupper);
        if (eviction_policy_str == "RANDOM") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::RANDOM;
        } else if (eviction_policy_str == "LRU" || eviction_policy_str == "TOLERANT") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::LRU;
        } else if (eviction_policy_str == "GUIDED") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::GUIDED;
        } else if (eviction_policy_str == "GUIDED_LRU") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::GUIDED_LRU;
        } else if (eviction_policy_str == "PERFECT_GUIDED") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::PERFECT_GUIDED;
        } else if (eviction_policy_str == "PERFECT_GUIDED_LRU") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::PERFECT_GUIDED_LRU;
        } else if (eviction_policy_str == "DEEPUM") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::DEEPUM;
        } else {
            wprintf("Defaulting eviction policy to be LRU\n", "");
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::LRU;
        }
        // migration policy
        std::transform(migration_policy_str.begin(), migration_policy_str.end(), migration_policy_str.begin(), ::toupper);
        if (migration_policy_str == "DEEPUM") {
            migration_policy = Simulator::MigPolicy::DEEPUM;
        } else {
            wprintf("Defaulting migration policy to be OURS\n", "");
            migration_policy = Simulator::MigPolicy::OURS;
        }
    }

    // parameter validation
    if (is_simulation) {
        SimulationParamSanityCheck();
    } else {
        if (is_input_pf_only) Assert(is_UVM);
    }
    // only one or less than one of these options are specified
    Assert(is_resnet + is_inception + is_senet <= 1);

    printf("End configs\n\n");

    // set random seed
    srand(0);

    bool output_folder_exists = system(("test -d " + output_folder_name).c_str()) == 0;
    if (output_folder_exists && !output_override) {
        wprintf("Output folder <%s> exists\n", output_folder_name.c_str());
    }

    // cout redirection
    RedirStdOut* r;

    ParseCommandLine(argc, argv);

    SetupOutputFolder();

    if (is_transformer==1)
    {
        transformer_parse(nn_model_input_file.c_str());
        transformer_op_datalow_pass(borden);
    }
    else {
    
        InitScanner();
        // yytokentype token;
        // while ((token = (yytokentype)yylex()) != 0)
        //     PrintOneToken(token, yytext, yylval, yylloc);
        InitParser();

        yyparse();

        layer_pre_pass_datasize();

        layer_first_pass_dataflow();
    }


    printf("\n");

    if (!is_simulation) {
        // tensor info
        r = new RedirStdOut("tensors.config");
        for (size_t i = 0; i < tensor_list.size(); i++) {
            tensor_list[i]->print();
        }
        delete r;
    }

   
    if (is_transformer==1)
    {
        // layer info
        r = new RedirStdOut("layers.config");
        for (size_t i = 0; i < forward_ops.size(); i++) {
            forward_ops[i]->print();
        }
        delete r;
        transformer_scheduling_kernels();
    }
    else
    {
        // layer info
        r = new RedirStdOut("layers.config");
        for (size_t i = 0; i < forward_layers.size(); i++) {
            forward_layers[i]->print();
        }
        delete r;

        layer_second_pass_scheduling_kernels();
    }

    // for (size_t i = 0; i < kernel_list.size(); i++) {
    //     kernel_list[i].print();
    // }

    nprintf("Global Memory amount: %lld B\n", memory_offset_weights);
    nprintf("Total Memory spend in 1 iteration: %lld B\n", memory_offset_intermediate);

    // return 0;

    printf("\n");
    if (is_simulation) {

        loadKernelTimes();

        loadWorkspaceSizes();
        // tensor info
        r = new RedirStdOut("tensors.config");
        for (size_t i = 0; i < tensor_list.size(); i++) {
            tensor_list[i]->print();
        }
        delete r;

        // kernel info
        r = new RedirStdOut("kernels.config");
        for (size_t i = 0; i < kernel_list.size(); i++) {
            kernel_list[i].print();
        }
        delete r;

        nprintf("Global Memory amount: %lld B\n", memory_offset_weights);
        nprintf("Total Memory spend in 1 iteration: %lld B\n", memory_offset_intermediate);


        nprintf("Memory Overcommitment: %lld B/%lld B, %f GB/%f GB (%f%%)\n", 
                memory_offset_intermediate + memory_offset_weights, (long long) (GPU_memory_size_GB * std::pow(1024, 3)),
                (memory_offset_intermediate + memory_offset_weights) / std::pow(1024, 3), GPU_memory_size_GB,
                (memory_offset_intermediate + memory_offset_weights) / (GPU_memory_size_GB * std::pow(1024, 3)) * 100);
        long max_num_pages = 0;
        CUDAKernel *max_mem_usage_kernel = nullptr;
        for (auto it = kernel_list.begin(); it != kernel_list.end(); ++it) {
            CUDAKernel *current_kernel = &(*it);
            vector<Tensor *> required_tensors;
            current_kernel->getRequiredTensors(required_tensors);
            long num_pages = 0;
            for (Tensor *tensor : required_tensors) {
                num_pages += std::ceil((float) tensor->size_in_byte / PAGE_SIZE);
            }
            if (num_pages > max_num_pages) {
                max_num_pages = num_pages;
                max_mem_usage_kernel = current_kernel;
            }
        }
        double max_memory_usage_GB = max_num_pages * PAGE_SIZE / std::pow(1024, 3);
        Assert(max_mem_usage_kernel != nullptr);
        nprintf("Memory Usage Maximized at Kernel%d: %lld B (%f GB)\n", 
                max_mem_usage_kernel->kernel_id, max_num_pages * PAGE_SIZE,
                max_memory_usage_GB);
        if (max_memory_usage_GB > GPU_memory_size_GB) {
            eprintf("Single kernel memory usage %f GB greater than total GPU memory size %f GB, aborting", 
                    max_memory_usage_GB, GPU_memory_size_GB);
            Assert(false);
        }

        
        
        tensor_first_pass_liveness_analysis();

        tensor_second_pass_interval_formation();

        get_interval_time();


        // life cycle info
        r = new RedirStdOut("interval.config");
        for (int i = 0; i < tensor_list.size(); i++) {
            tensor_list[i]->print_liveness();
            tensor_list[i]->print_intervals();
        }
        delete r;


        give_eviction_guide();
        
        // r = new RedirStdOut("evc_guide_compressed.config");
        // int max_len = 0, max_idx = -1;
        // std::map<int, int> distri;
        // for (int tensor_idx = 0; tensor_idx < tensor_list.size(); tensor_idx++) {
        //     Tensor *candidate = tensor_list[tensor_idx];
        //     int cur_len = 0;
        //     std::cout << "Tensor: " << candidate->tensor_id << "\n";
        //     Eviction_P current_hotness = EvictionGuide_Table[0].entry[candidate];
        //     for (long i = 0; i < kernel_list.size(); i++) {
        //         Eviction_P hotness = EvictionGuide_Table[i].entry[candidate];
        //         if (i == 0 || hotness != current_hotness) {
        //             std::cout << i << ":" << print_eviction_array[hotness].c_str() << "\n";
        //             current_hotness = hotness;
        //             cur_len++;
        //         }
        //     }
        //     if (cur_len > max_len) {
        //         max_len = cur_len;
        //         max_idx = tensor_idx;
        //     }
        //     distri[cur_len]++;
        //     std::cout << "\n";
        // }
        // std::cout << "Max len: " << max_len << " @ Tensor: " << max_idx << "\n";
        // std::cout << "Distribution:\n";
        // for (auto it = distri.begin(); it != distri.end(); ++it) {
        //     std::cout << "  " << it->first << ":" << it->second << "\n";
        // }
        // delete r;

        
        //Implementation of flashneuron
        if (migration_policy_str=="FLASHNEURON"|| migration_policy_str=="G10GDSSSD" || migration_policy_str=="G10GDSFULL")
        {
            if (migration_policy_str=="FLASHNEURON")
            {
                int fail;
                fail = scheduling_offload_flashneuron();
                if (fail == 1)
                {
                    std::cout<<"@@@ Flashneuron cannot support this large model!"<<std::endl;
                    return 0;
                }
                print_offloading_flashneuron();
                std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"<<std::endl;
            }
            else
            {
                r = new RedirStdOut("pre_dealloc.config");
                scheduling_prefetch();
                delete r;
                // prefetch guide
                r = new RedirStdOut("prefetch_guide.config");
                print_prefetch_table();
                delete r;
            }
            

            GDS_Baseline_Type sim_type;
            if (migration_policy_str=="FLASHNEURON")
            {
                sim_type = GDS_Baseline_Type::FlashNeuron;
            }
            else if (migration_policy_str=="G10GDSSSD")
            {
                sim_type = GDS_Baseline_Type::G10_GDS_SSD;
            }
            else
            {
                sim_type = GDS_Baseline_Type::G10_GDS_FULL;
            }
        
            FlashNeuron_simulator sim(SSD_PCIe_bandwidth_GBps, CPU_PCIe_bandwidth_GBps, GPU_memory_size_GB, sim_type);
            sim.run();
            double time_ = sim.total_sim_time;
            std::string info_file = output_folder_name + "/sim_result.final";
            std::ofstream foout(info_file);
            foout<<"total_exe_time = "<<time_<<std::endl; 
            foout<<"total_time_breakdown_stall = "<<sim.total_time_breakdown_stall<<std::endl;
            foout<<"total_time_breakdown_overlap = "<<sim.total_time_breakdown_overlap<<std::endl;
            foout<<"total_time_breakdown_executionOnly = "<<sim.total_time_breakdown_exe<<std::endl;
            foout<<"total_ssd2gpu_byte = "<<sim.total_fetch_byte<<std::endl;
            foout<<"total_gpu2ssd_byte = "<<sim.total_offload_byte<<std::endl;
            std::string info_file2 = output_folder_name + "/sim_result.kernelStall";
            std::ofstream fooout(info_file2);
            if (migration_policy_str=="FLASHNEURON")
            {
                for (int i = 0; i < sim.fl_kernel_stall_normed.size(); i++)
                {
                    fooout<<(sim.fl_kernel_stall_normed[i] < 0.0001 ? 0 : sim.fl_kernel_stall_normed[i]) <<std::endl;
                }
            }
            
            
            return 0;
        }

        // eviction guide
        // r = new RedirStdOut("evc_guide.config");
        // print_eviction_guide_table();
        // delete r;

        r = new RedirStdOut("pre_dealloc.config");
        scheduling_prefetch();
        delete r;

        // prefetch guide
        r = new RedirStdOut("prefetch_guide.config");
        print_prefetch_table();
        delete r;

        

        // real memory usage
        r = new RedirStdOut("real_mem.config");
        print_GPU_mem_really_in_use();
        delete r;

        // kernel time table
        r = new RedirStdOut("kernel_time_table.config");
        for (int i = 0; i < kernel_list.size(); i++) {
            std::cout << kernel_time_table[i] << std::endl;
        }
        delete r;





/***********************************Getting Motivation Number***************************************/
        string argv1 = output_folder_name;
        string filenaeme;
        filenaeme = argv1+"_NNMemConsumptionLog.py";
        printf("%s\n", filenaeme.c_str());
        std::ofstream motiv_1(filenaeme);

        motiv_1<<"active = [";
        for (auto it = kernel_list.begin(); it != kernel_list.end(); ++it) {
            CUDAKernel *current_kernel = &(*it);
            vector<Tensor *> required_tensors;
            current_kernel->getRequiredTensors(required_tensors);
            long num_bytes = 0;
            for (Tensor *tensor : required_tensors) {
                num_bytes += std::ceil((float) tensor->size_in_byte);
            }
            motiv_1<<num_bytes<<",";
        }

        motiv_1<<"]\n";

        motiv_1 << "active_breakdown = [";
        for (auto it = kernel_list.begin(); it != kernel_list.end(); ++it) {
            CUDAKernel *current_kernel = &(*it);
            vector<Tensor *> inputs, weights, intermediates;
            current_kernel->getTensorBreakdown(inputs, weights, intermediates);
            long input_bytes = 0, weight_bytes = 0, intermediate_bytes = 0;
            for (Tensor *tensor : inputs)
                input_bytes += std::ceil((float) tensor->size_in_byte);
            for (Tensor *tensor : weights)
                weight_bytes += std::ceil((float) tensor->size_in_byte);
            for (Tensor *tensor : intermediates)
                intermediate_bytes += std::ceil((float) tensor->size_in_byte);
            motiv_1<< "(" << input_bytes << "," << weight_bytes << "," << intermediate_bytes << "),";
        }

        motiv_1<<"]\n";


        std::vector<long> GPU_pressure_memory_estimation;
        GPU_pressure_memory_estimation.resize(kernel_list.size());
        long total_global_size;
        for (int i = 0; i < kernel_list.size(); i++)
        {
            GPU_pressure_memory_estimation[i] = memory_offset_intermediate + memory_offset_weights + tensor_list[0]->size_in_byte;
        }
        for (int i = 0; i < tensor_list.size(); i++)
        {
            if (!tensor_list[i]->is_global_weight)
            {
                for (int j = 0; j < tensor_list[i]->live_interval[0]; j++)
                {
                    GPU_pressure_memory_estimation[j] -= tensor_list[i]->size_in_byte;
                }
                int death;
                if (tensor_list[i]->live_interval[1]==-1)
                {
                    death = tensor_list[i]->live_interval[0]+1;
                }else
                {
                    death = tensor_list[i]->live_interval[1];
                }
                
                for (int j = death; j < kernel_list.size(); j++)
                {
                    GPU_pressure_memory_estimation[j] -= tensor_list[i]->size_in_byte;
                }   
            }
        }

        motiv_1 << "total = [";
        for (int i = 0; i < kernel_list.size(); i++)
        {
            motiv_1<<GPU_pressure_memory_estimation[i]<<",";
        }
        motiv_1<<"]\n";

        motiv_1 << "global_weight = " << memory_offset_weights << "\n";
        motiv_1 << "input_size = " << tensor_list[0]->size_in_byte << "\n";
        
        motiv_1.close();

        filenaeme = argv1+"_TensorPeriodLog.py";
        std::ofstream motiv_2(filenaeme);
        motiv_2 << "sd_size = [";
        for (int i = 0; i < interval_list.size(); i++)
        {
            motiv_2<<interval_list[i]->the_tensor->size_in_byte;
            motiv_2<<", ";
        }
        motiv_2<<"]\n";
        motiv_2 << "sd_time = [";
        for (int i = 0; i < interval_list.size(); i++)
        {
            motiv_2<<interval_list[i]->time_estimated;
            motiv_2<<", ";
        }
        motiv_2<<"]\n";
        motiv_2 << "# ";
        for (int i = 0; i < interval_list.size(); i++)
        {
            motiv_2<<interval_list[i]->kernelLevel_interval[0];
            motiv_2<<" ";
        }


        motiv_2.close();
        
        
/***********************************Getting Motivation Number   End***************************************/


        nprintf("Average interval time: %f ms\n\n", 
                interval_list[(interval_list.size() - 1) / 2]->time_estimated);
        
        iprintf("Checking output stat files\n", "");
        Simulator::Stat stat(stat_output_file);
        if (!stat.outputFileExists()) {
            if (kernel_time_std_dev != 0) {
                printf("Kernel time variation with std %f\n", kernel_time_std_dev);
                std::uniform_real_distribution<double> distribution(1 - kernel_time_std_dev, 1 + kernel_time_std_dev);
                if (ran_seed != 1)
                {
                    rand_device.seed((unsigned int)(ran_seed));
                }
                // rand_device.seed((unsigned int)(100*kernel_time_std_dev));
                for (int i = 0; i < kernel_list.size(); i++) {
                    double ratio = distribution(rand_device);
                    if (ratio < 0.1) ratio = 0.1; // prevent normal distribution to produce a negative number
                    if (ratio > 1.9) ratio = 1.9; // ensure that the mean is still around 1.0
                    kernel_list[i].execution_cycles *= ratio;
                    kernel_list[i].input_pf_execution_cycles *= ratio;
                    kernel_list[i].pf_execution_cycles *= ratio;
                    Assert(kernel_list[i].execution_cycles > 0);
                    Assert(kernel_list[i].input_pf_execution_cycles > 0);
                    Assert(kernel_list[i].pf_execution_cycles > 0);
                }
            }
            iprintf("\nSimulation\n", "");
            Simulator::EventSimulator *sim = new Simulator::EventSimulator(stat_output_file);
            sim->run(num_iteration);
            delete sim; // make sure stats are written back to the files
        }
        iprintf("\nAnalysis\n", "");
        stat.prepareOutputFiles(true);
        stat.analyzeStat();
    } else {
        if (is_cudnn) {
            iprintf("Generating main code -- CUDNN mode\n", "");
            auto start_time = high_resolution_clock::now();
            // cudnn_profiling(true);          // normal run, individual
            cudnn_profiling(false);         // normal run, grouped
            // cudnn_profiling(false, true);   // workspace only
            duration<float> fsec = high_resolution_clock::now() - start_time;
            iprintf("Profiling duration: %fs (%fms)\n", fsec.count(), fsec.count() * 1000);
        } else {
            wprintf("Profiling without CUDNN deprecated\n", "");
            iprintf("Generating main code -- %s mode\n", is_individual ? "individual" : "whole");
            main_code_generation();
            printf("\n");

            if (is_compile || is_run) {
                printf("Profiling with Individual: %s, Compile: %s, Run: %s\n",
                    is_individual ? "True" : "False",
                    is_compile ? "True" : "False",
                    is_run ? "True" : "False");
                // run profiling scripts
                std::string args = " ";
                if (is_compile)
                    args += "-c ";
                if (is_compile && compile_max_thread_num > 0)
                    args += "-t " + to_string(compile_max_thread_num) + " ";
                if (is_run)
                    args += "-r ";
                Assert(system((output_folder_name + "/scripts/compileAndRun.sh" + args).c_str()) == 0);
            } else {
                iprintf("Both Compile and Run are disabled, run <%s> manually to profile\n",
                    (output_folder_name + "/scripts/compileAndRun.sh").c_str());
            }
        }
    }

    for (int i = 0; i < forward_layers.size(); i++)
    {
      delete forward_layers[i];
    }
    for (int i = 0; i < tensor_list.size(); i++)
    {
      delete tensor_list[i];
    }


    return (ReportError::NumErrors() == 0? 0 : -1);
}
