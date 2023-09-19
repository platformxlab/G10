#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

/*
The simulator of G10 is part ly based on the UVMSmart and GPGPUSim simulators. (Including UVM migration latencies,
PCIe characteristics, Page replacement policies, far-fault managements, etc.)
https://github.com/DebashisGanguly/gpgpu-sim_UVMSmart
*/

#include <vector>
#include <deque>
#include <queue>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include "simulationComponents.h"
#include "simulationEvents.h"
#include "simulationUtils.h"
#include "analysis.h"
#include "ast.h"

using std::deque;
using std::string;
using std::vector;
using std::greater;
using std::unordered_map;
using std::unordered_set;
using std::priority_queue;

#define PAGE_SIZE (4096)


namespace Simulator {

/**
 * @brief Event simulator that operates on a message queue and process events on a time
 *        basis
 */
class EventSimulator {
  public:
    /**
     * @brief create a new event simulator
     * @param basename the output file that will be using to store the result
     * @note <stat_output_file>.kernel -- kernel related results
     *       <stat_output_file>.pcie   -- PCIe related results
     */
    EventSimulator(string basename);
    ~EventSimulator();

    /**
     * @brief step the simulation forward by parsing next event
     */
    void step();

    /**
     * @brief run one iteration (from the first kernel to last kernel) of simulation
     */
    void run();

    /**
     * @brief run several iterations of simulation
     * @param num_iter desired number of iteration to run
     */
    void run(unsigned num_iter);

  private:
    /**
     * @brief add an event to the event queue
     * @param event the event to be added
     */
    void schedule(Event *event);

    /**
     * @brief add a list of events to the event queue
     * @param events the list of events to be added
     */
    void schedule(vector<Event *> &events);

    // event queue
    priority_queue<EventPtr, vector<EventPtr>, EventPtrComp> event_queue;
    // simulation current time, updated before the actual events are run
    unsigned long current_time;
};

} // namespace Simulator

// class Simulation {
//   private:
//     class EventQueue {

//       unsigned long current_timestamp;
//     };

//   public:
//     /**
//      * @brief Initialize simulation
//      * @param parameters all parameters are passed in by global variable
//      */
//     Simulation();

//     /**
//      * @brief Simulate the whole system
//      * @note The time unit used in simulation is the execution time of each of the
//      *       kernel. All the events are handled between executing of two kernels
//      */
//     void simulate();

//   private:
//     // Page tables
//     /**
//      * @brief CPU page table entry layout
//      */
//     class CPUPageTableEntry {
//       public:
//         Addr ppn;
//         PageLocation location;
//         PageStatus in_transfer;
//     };
//     unordered_map<Addr, CPUPageTableEntry> CPU_page_table;
//     unordered_set<Addr> CPU_phys_page_avail;

//     // alloc a PTE for entry stored in host memory, resize the CPU max buffer if needed
//     void alloc_CPU_PTE(Addr vpn);
//     // mark CPU PTE in transfer
//     void mark_invalid_CPU_PTE(Addr vpn);
//     // erase the PTE for entry stored in host memory
//     void erase_CPU_PTE(Addr vpn);

//     void reportCPUPageTable();

//     /**
//      * @brief GPU page table entry layout
//      */
//     class GPUPageTableEntry {
//       public:
//         GPUPageTableEntry() : ppn(0), in_transfer(PRESENT) {};
//         GPUPageTableEntry(Addr ppn) : ppn(ppn), in_transfer(PRESENT) {};
//         Addr ppn;
//         PageStatus in_transfer;
//       private:
//         Tensor *tensor;
//     };
//     unordered_map<Addr, GPUPageTableEntry> GPU_page_table;
//     unordered_set<Addr> GPU_phys_page_avail;

//     // alloc a PTE for entry stored in GPU
//     bool alloc_GPU_PTE(Addr vpn);
//     // remove a PTE for entry stored in GPU
//     void free_GPU_PTE(Addr vpn);
//     // mark GPU PTE in transfer
//     void mark_invalid_GPU_PTE(Addr vpn);
//     // erase the PTE for entry stored in host memory
//     void erase_GPU_PTE(Addr vpn);
//     // return vpn to the entry that are desired to be removed
//     Addr evict_GPU_PTE();

//     void reportGPUPageTable();

//     // PCIe functionalities
//     class PCIeQueue {
//       public:
//         PCIeQueue(unsigned latency_cycle, string name) :
//             latency_cycle(latency_cycle), name(name);
//         void addToFront(Addr page_start);
//         void addToBack(Addr page_start);
//         // if
//         bool needTransfer();
//         // move one page in the waiting queue to the in transfer queue
//         Addr transferFrontPage(unsigned long timestamp);
//         // check if there are pages that completed transferring
//         bool hasTransferCompletePage(unsigned long timestamp);
//         // check if there are still critical pages in the queues
//         bool hasTransferringCriticalPage();
//         // check if there are still critical pages in the waiting queue
//         bool hasWaitingCriticalPage();
//         // get the first completed page, will fail if there are no such entry
//         Addr getFrontCompletePage(unsigned long timestamp);
//         // get
//         unsigned long getCriticalCompleteTime();
//         void removeFrontCompletePage(unsigned long timestamp);
//         void print();
//         void reportQueueStatus();

//         const unsigned latency_cycle;
//       private:
//         PCIeQueue();

//         deque<Addr> in_transfer_queue;
//         deque<Addr> waiting_queue;
//         unordered_map<Addr, long> metadata;
//         unordered_set<Addr> critical_pages;
//         string name;

//         unsigned long transferred_pages_history = 0;
//     };

//     class GPUMMU {
//       public:
//         enum MMUAction { ALLOC, FREE };
//         struct GPUMMUEntry {
//           Addr page_start;
//           unsigned long size;
//           MMUAction action;
//           bool is_critical;
//           unsigned long finishing_timestamp;
//         };

//         GPUMMU(unsigned alloc_lat_slp, unsigned alloc_lat_off,
//                unsigned free_lat_slp, unsigned free_lat_off) :
//             alloc_lat_slp(alloc_lat_slp), alloc_lat_off(alloc_lat_off),
//             free_lat_slp(free_lat_slp), free_lat_off(free_lat_off) {};
//         void addToQueue(Addr page_start, unsigned long size,
//                         MMUAction action, bool critical);
//         bool needAction();
//         GPUMMUEntry manageFrontRequest(unsigned long timestamp);
//         bool hasCompletedRequest(unsigned long timestamp);
//         bool hasManagingCriticalRequest();
//         bool hasWaitingCriticalRequest();
//         bool getFrontCompletedRequest(unsigned long timestamp,
//                                       GPUMMUEntry &entry);
//         bool removeFrontCompletedRequest(unsigned long timestamp);

//         void reportMMUStatus();

//       private:
//         GPUMMU();

//         unsigned long getLatencyCycles(GPUMMUEntry &entry);
//         const unsigned alloc_lat_slp;
//         const unsigned alloc_lat_off;
//         const unsigned free_lat_slp;
//         const unsigned free_lat_off;

//         deque<GPUMMUEntry> managing_queue;
//         deque<GPUMMUEntry> waiting_queue;
//         size_t critical_pages_count;
//         string name;

//         unsigned long alloc_request_num = 0;
//         unsigned long alloc_pages = 0;
//         unsigned long free_request_num = 0;
//         unsigned long free_pages = 0;
//     };

//     bool processCompleteEntryInGPUQueue(PCIeQueue &queue, unsigned long current_time);
//     void processCompleteEntryInCPUQueue(unsigned long current_time);
//     void processCompleteEntryInSSDQueue(unsigned long current_time);

//     void processGPUPCIeQueues(unsigned long delta_cycles);

//     void processGPUMemoryReqQueues(unsigned long delta_cycles);

//     unsigned long GPUGetWaitingCriticalStepCycles();

//     unsigned long getGPUCriticalCompleteDeltaCycle();

//     /**
//      * @brief Consumes data movement that have GPU as it destination
//      *
//      * @param timestamp current timestamp, in cycles
//      * @return pages that are actually batched
//      */
//     void GPUConsumeMallocRequest(unsigned long timestamp);

//     /**
//      * @brief Consumes data movement that have GPU as it destination
//      *
//      * @param timestamp current timestamp, in cycles
//      * @return pages that are actually batched
//      */
//     void GPUConsumeIncomingMovement(unsigned long timestamp);

//     /**
//      * @brief Consumes data movement that have GPU as it source
//      *
//      * @param timestamp current timestamp, in cycles
//      */
//     void GPUHandleOutgoingMovement(unsigned long timestamp);

//     bool requiredPageArrived(vector<Tensor *> &required_tensors);

//     void reportTensorStatus(vector<Tensor *> &required_tensors);
//     void reportAllQueueStatus();

//     class PageFaultInfo {
//       public:
//         PageFaultInfo() : not_presented_pages(0),
//             CPU_to_GPU_faulted_pages(0), SSD_to_GPU_faulted_pages(0),
//             kernel(nullptr) {}
//         PageFaultInfo &operator+=(const PageFaultInfo &rhs) {
//           Assert(kernel == rhs.kernel);
//           not_presented_pages += rhs.not_presented_pages;
//           CPU_to_GPU_faulted_pages += rhs.CPU_to_GPU_faulted_pages;
//           SSD_to_GPU_faulted_pages += rhs.SSD_to_GPU_faulted_pages;
//           return *this;
//         }
//         unsigned long not_presented_pages;
//         unsigned long CPU_to_GPU_faulted_pages;
//         unsigned long SSD_to_GPU_faulted_pages;
//         CUDAKernel *kernel;
//     };
//     PageFaultInfo transferPagesForTensor(Tensor *tensor, bool is_critical);

//     unsigned long getPageFaultTime(PageFaultInfo &info);

//     // parameters
//     /**
//      * @brief Frequency of GPU in Hz
//      */
//     const unsigned GPU_frequency_Hz;

//     /**
//      * @brief Total memory size of GPU, in pages
//      */
//     const long GPU_total_memory_pages;

//     /**
//      * @brief PCIe latency in cycles
//      */
//     const unsigned PCIe_latency_cycles;

//     /**
//      * @brief PCIe bandwidth in byte per cycle
//      */
//     const double PCIe_bandwidth_Bpc;

//     const unsigned PCIe_batch_size_page;
//     /**
//      * @brief PCIe batch initiation interval
//      */
//     const unsigned PCIe_batch_ii_cycle;

//     /**
//      * @brief time taken for GPU to malloc a page in cycles
//      */
//     const unsigned GPU_malloc_cycle_per_page;

//     /**
//      * @brief time taken for GPU to free a page in cycles
//      */
//     const unsigned GPU_free_cycle_per_page;

//     const unsigned SSD_read_latency_cycle;
//     const unsigned SSD_write_latency_cycle;


//     const GPUPageTableEvcPolicy evc_policy;
//     // parameters END

//     // workspace
//     /**
//      * @brief Total memory size used on CPU, in pages
//      */
//     long CPU_total_memory_pages;

//     /**
//      * @brief Number of cycles that not used by the pervious iteration of the
//      *        PCIe transfer because it cannot create a further batch. These
//      *        cycles can be added to the next round of simulation
//      *
//      */
//     unsigned long GPU_PCIe_remaining_cycles;
//     PageLocation PCIe_GPU_in_last_taken;
//     PageLocation PCIe_GPU_out_last_taken;

//     unsigned long GPU_malloc_remaining_cycles;
//     unsigned long GPU_free_remaining_cycles;

//     /**
//      * @brief Current simulation cycle
//      */
//     unsigned long current_cycle;

//     // PCIe and Message queues
//     PCIeQueue SSD_to_GPU_PCIe_queue;
//     PCIeQueue CPU_to_GPU_PCIe_queue;
//     PCIeQueue GPU_to_CPU_PCIe_queue;
//     PCIeQueue GPU_to_SSD_PCIe_queue;
//     MessageQueue GPU_MMU_queue;
//     // workspace END
// };

#endif
