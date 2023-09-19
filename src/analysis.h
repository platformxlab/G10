#ifndef __ANALYSIS_H__
#define __ANALYSIS_H__

#include "ast.h"
#include "simulationUtils.h"
#include <string>
#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <queue>


typedef enum {
    Conv2d_Forward, ReLU_Forward, MaxPool2d_Forward, AdaptiveAvgPool2d_Forward, Linear_Forward, 
    Dropout_Forward, BatchNorm2d_Forward, Conv2d_Backward_Weight, Conv2d_Backward_Input, Conv2d_Apply_Grad,
    ReLU_Backward, MaxPool2d_Backward, AdaptiveAvgPool2d_Backward, Linear_Backward_Weight, Linear_Backward_Input, 
    Linear_Backward_Bias, Linear_Apply_Grad_Bias, Linear_Apply_Grad_Weight, Dropout_Backward, BatchNorm2d_Backward,
    BatchNorm2d_Apply_Grad, LoadData_A0, makeLoss, Add_Forward, Add_MultiGredient, Concat_Forward, Concat_Backward,
    Scale_Forward, Scale_Backward, GatherV2_Forward, GatherV2_Backward, Add_Backward, Divide_Forward, Divide_Backward_A, Divide_Backward_B, 
    Multiply_Forward, Multiply_Backward, Power_Forward, Power_Backward, Sqrt_Forward, Sqrt_Backward, SoftmaxBasic_Forward, 
    SoftmaxBasic_Backward, Subtract_Forward, Subtract_Backward, Sum_Forward, Sum_Backward, Tanh_Forward, Tanh_Backward, 
    BatchMatMul_Forward, BatchMatMul_Backward, Apply_Grad, Erf_Forward, Erf_Backward
} CUDAKernelType;

const std::string print_kerneltype_array [54] = {
    "Conv2d_Forward", "ReLU_Forward", "MaxPool2d_Forward", "AdaptiveAvgPool2d_Forward", "Linear_Forward", 
    "Dropout_Forward", "BatchNorm2d_Forward", "Conv2d_Backward_Weight", "Conv2d_Backward_Input", "Conv2d_Apply_Grad",
    "ReLU_Backward", "MaxPool2d_Backward", "AdaptiveAvgPool2d_Backward", "Linear_Backward_Weight", "Linear_Backward_Input", 
    "Linear_Backward_Bias", "Linear_Apply_Grad_Bias", "Linear_Apply_Grad_Weight", "Dropout_Backward", "BatchNorm2d_Backward",
    "BatchNorm2d_Apply_Grad", "LoadData_A0", "makeLoss", "Add_Forward", "Add_MultiGredient", "Concat_Forward", "Concat_Backward",
    "Scale_Forward", "Scale_Backward", "GatherV2_Forward", "GatherV2_Backward", "Add_Backward", "Divide_Forward", "Divide_Backward_A", "Divide_Backward_B", 
    "Multiply_Forward", "Multiply_Backward", "Power_Forward", "Power_Backward", "Sqrt_Forward", "Sqrt_Backward", "SoftmaxBasic_Forward", 
    "SoftmaxBasic_Backward", "Subtract_Forward", "Subtract_Backward", "Sum_Forward", "Sum_Backward", "Tanh_Forward", "Tanh_Backward", 
    "BatchMatMul_Forward", "BatchMatMul_Backward", "Apply_Grad", "Erf_Forward", "Erf_Backward"
};

enum Eviction_P {
    Hot, Medium, Cold, Dead
};

const std::string print_eviction_array [4] = {
    "hot", "medium", "cold", "dead"
};


class CUDAKernel {
    public:
        int kernel_id;
        CUDAKernelType type;
        Model_Layer* parent_layer = nullptr;
        Model_OP* parent_op = nullptr;
        std::unordered_set<Tensor*> inputs;
        std::unordered_set<Tensor*> outputs;
        Tensor* workspace = nullptr;
        
        /**
         * @brief number of cycles for the kernel to execute assume all the tensors 
         * are presented in the GPU memory and ready for computation.
         */
        long execution_cycles = -1;
        long pf_execution_cycles = -1;
        long input_pf_execution_cycles = -1;

        CUDAKernel(CUDAKernelType t, Model_Layer* layer);
        CUDAKernel(CUDAKernelType t, Model_OP* op_layer);
        void getRequiredTensors(std::vector<Tensor*> &required_tensors) const;
        void getRequiredTensors(std::unordered_set<Tensor*> &required_tensors) const;
        void getRequiredTensors(std::vector<Tensor*> &required_tensors,
                                std::vector<Tensor*> &required_input_tensors,
                                std::vector<Tensor*> &required_output_tensors) const;
        void getTensorBreakdown(std::vector<Tensor*> &inputs,
                                std::vector<Tensor*> &weights,
                                std::vector<Tensor*> &intermediates) const;
        void print();
};


class EvictionGuide_Entry {
    public:
        std::unordered_map<Tensor*, Eviction_P> entry;
        std::unordered_map<Tensor*, double> absolute_time_entry;
};


class Offload_Hint_FlashNeuron {
  public:
    Offload_Hint_FlashNeuron(int issued_time, Tensor* tensor) :
        issued_time(issued_time), tensor(tensor) {}

    bool operator<(const Offload_Hint_FlashNeuron& rhs) const {
      return issued_time < rhs.issued_time;
    }

    int issued_time;
    Tensor* tensor;
};


struct flashneuron_PTE{
    bool valid;
};

class FlashNeuron_memory_manager{
    private:
        std::vector<flashneuron_PTE> page_table;
    public:
        FlashNeuron_memory_manager(double GPUsize_GB);
        int alloc_from_left(Tensor* tensor);
        int alloc_from_right(Tensor* tensor);
        long largest_available_size();
        void dealloc_tensor(Tensor* tensor);
        double util_cal();
};


typedef enum {
    FlashNeuron, G10_GDS_SSD, G10_GDS_FULL
} GDS_Baseline_Type;

typedef enum {
    Offload_Finish, Prefetch_Finish, Kernel_Finish, Offload_Finish_CPU, Prefetch_Finish_CPU
} Fl_Pending_Event_Type;


struct fl_pending_event{
    Fl_Pending_Event_Type type;
    double ready_time;
    long event_id;
};

struct Fl_event_less
{
    bool operator() (const fl_pending_event a, const fl_pending_event b) const {
        return (a.ready_time > b.ready_time);
    }
};

// auto Fl_event_less = [](fl_pending_event a, fl_pending_event b) { return a.ready_time > b.ready_time; };

struct fetch_wait
{
    Tensor* tensor;
    Simulator::PageLocation source;
};



struct fl_fetch{
    Tensor* tensor;
    long event_id;
    double estimated_time;
    bool is_happening;
};

struct fl_offload{
    Tensor* tensor;
    long event_id;
    bool is_happening;
    double estimated_time;
};

class FlashNeuron_simulator{
    public:
        double total_sim_time; // ms
        double total_trasfer_time;
        double total_time_breakdown_stall;
        double total_time_breakdown_overlap;
        double total_time_breakdown_exe;
        long total_offload_byte;
        long total_fetch_byte;
        double BW_ssd; // B/ms
        double BW_pcie; // B/ms
        long event_number = 0;
        GDS_Baseline_Type baseline_type;
        std::queue<fl_fetch> fl_fetch_queue;
        std::queue<fl_fetch> fl_fetch_queue_cpu;
        std::vector<double> fl_kernel_stall_normed;
        std::deque<fl_offload> fl_offload_queue;
        std::deque<fl_offload> fl_offload_queue_cpu;
        std::queue<fetch_wait> fetch_allocate_waiting_queue;
        std::set<Tensor*> fetch_allocate_waiting_tensors;
        std::set<Tensor*> cpu_tensors;
        FlashNeuron_memory_manager mem_manager;
        FlashNeuron_simulator(double BW_ssd_GBs, double BW_pcie_GBs, double GPU_size_GB, GDS_Baseline_Type type);
        void run();

        // return 0 for success, return 1 for failure, return 2 for stalled by kernel
        int serve_one_pending_event(int kernel_event_id);
        void check_fetch_allocation();
};

//Transformers:
void transformer_op_datalow_pass(int borden);



void layer_pre_pass_datasize();

void layer_first_pass_dataflow();

void layer_second_pass_scheduling_kernels();

void transformer_scheduling_kernels();

void tensor_first_pass_liveness_analysis();

void tensor_second_pass_interval_formation();

void get_interval_time();

void give_eviction_guide();

void print_eviction_guide_table();

void scheduling_prefetch();

int scheduling_offload_flashneuron();

void print_offloading_flashneuron();

double simulate_flashneuron();

void print_prefetch_table();

void print_GPU_mem_estimation();

void print_GPU_mem_really_in_use();



























#endif