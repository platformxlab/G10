#include <math.h>
#include <fstream>
#include <string>
#include <cstring>
#include <random>
#include <algorithm>
#include <math.h>
#include <climits>
#include <iostream>
#include <unistd.h>
#include "simulationComponents.h"
#include "simulationEvents.h"
#include "simulator.h"
#include "printUtils.h"

using std::pow;
using std::ceil;
using std::sort;
using std::pair;
using std::round;
using std::string;
using std::vector;
using std::ofstream;
using std::unordered_set;
using Simulator::DataMovementHint;
using Simulator::PageLocation;

extern vector<Tensor*> tensor_list;
extern vector<CUDAKernel> kernel_list;
extern vector<Model_Layer*> forward_layers;
extern vector<DataMovementHint> movement_hints;

extern long long memory_offset_intermediate;
extern long long memory_offset_weights;

extern int is_UVM;

namespace Simulator {

extern System *sim_sys;
extern Stat *sim_stat;

EventSimulator::EventSimulator(string basename) {
  sim_sys = new Simulator::System();
  sim_stat = new Simulator::Stat(basename);
  sim_stat->prepareOutputFiles();

  // Assert(memory_offset_intermediate < (long long) LONG_MAX);
  // Assert(memory_offset_weights < (long long) LONG_MAX);
  // Assert(memory_offset_intermediate + memory_offset_weights < (long long) LONG_MAX);

  // Assert(isPageAligned(memory_offset_intermediate));
  // Assert(isPageAligned(memory_offset_weights));

  long ideal_exe_time = 0;
  for (CUDAKernel kernel : kernel_list)
    ideal_exe_time += kernel.execution_cycles;
  printf("Ideal Execution Time: %ld cycles\n", ideal_exe_time);

  // Initialize CPU page table
  printf("Initializing CPU Page Table\n");
  for (Tensor *tensor : tensor_list) {
    Addr starting_addr = tensor->getGlobalOffset();
    long size_in_byte = (long) tensor->size_in_byte;
    long total_pages = ceil((double) size_in_byte / PAGE_SIZE);
    for (long page_num = 0; page_num < total_pages; page_num++) {
      Addr page_starting_addr = starting_addr + PAGE_SIZE * page_num;
      // create and get entry
      CPUPageTable::CPUPageTableEntry *entry =
          sim_sys->CPU_PT.createEntry(page_starting_addr);
      // Assert(entry);
      entry->ppn = 0;
      if (is_UVM) {
        if (tensor->is_global_weight) {
          entry->location = IN_SSD; // UVM enabled, tensor started in SSD
        } else {
          entry->location = NOT_PRESENT; // UVM enabled, tensor started unallocated
        }
      } else {
        entry->location = IN_GPU; // ideal case, all tensor in GPU
      }
      entry->in_transfer = false;
    }
  }
  current_time = 0;

  // push initial events to event queue
  printf("Initial Events\n");
  vector<Event *> initial_events;
  initial_events.push_back(new KernelBeginEvent(current_time, sim_sys->getCurrentKernel()));
  initial_events.push_back(new BatcherEvent(current_time));
  schedule(initial_events);
  printf("Initial Events END #<%ld>\n", event_queue.size());
  printf("Simulation Start =============================================================\n");
}

EventSimulator::~EventSimulator() {
  sim_stat->printSizeInfo();
  delete sim_sys;
  delete sim_stat;
}

  // if (strncmp(event->name.c_str(), "Kernel", 6) == 0)
void EventSimulator::step() {
  vector<Event *> created_events;
  // get currently scheduled event
  Event *scheduled_event = event_queue.top().ptr;
  // advance time
  // if (current_time != scheduled_event->scheduled_time) {
  //   printf("Time advanced from <%15ld> to <%15ld> ===================\n",
  //       current_time, scheduled_event->scheduled_time);
  // }
  // Assert(current_time <= scheduled_event->scheduled_time);
  current_time = scheduled_event->scheduled_time;
  // check if event is should be executed
  if (scheduled_event->shouldExecute()) {
    if (dynamic_cast<KernelBeginEvent *>(scheduled_event))
      printf("Executing: %s @ %ld [Duration:%ld] [ITER: %d] TotEvt:%ld\n",
          scheduled_event->name.c_str(), current_time,
          dynamic_cast<KernelBeginEvent *>(scheduled_event)->kernel->execution_cycles,
          sim_sys->getCurrentIteration(),
          event_queue.size());
    // execute current event
    scheduled_event->execute(created_events);
    // pop current event from event queue and create new events
    event_queue.pop();
    schedule(created_events);
    delete scheduled_event;
  }
}

void EventSimulator::run() {
  int this_iter = sim_sys->getCurrentIteration();
  iprintf("Simulation [ITER: %d] run starts @ %ld\n", this_iter, current_time);
  while (sim_sys->getCurrentIteration() == this_iter &&
         event_queue.size() > 0) {
    step();
  }
  iprintf("Simulation [ITER: %d] run ends @ %ld\n", this_iter, current_time);
}

void EventSimulator::run(unsigned num_iter) {
  printf("Simulation <%d> runs scheduled\n", num_iter);
  for (unsigned iter = 0; iter < num_iter; iter++)
    run();
}

void EventSimulator::schedule(Event *event) {
  if (event->name != "Exec")
  printf("  Scheduling: %s @ %ld\n",
      event->name.c_str(), event->scheduled_time);
  event_queue.emplace(EventPtr(event));
}

void EventSimulator::schedule(vector<Event *> &events) {
  for (Event *event : events)
    schedule(event);
}

} // namespace Simulator

// Simulation::Simulation() {

//   sort(movement_hints.begin(), movement_hints.end());

//   // initialization
//   CPU_phys_page_avail.reserve(CPU_total_memory_pages);
//   for (size_t current_page = 0; current_page < CPU_total_memory_pages; current_page++) {
//     CPU_phys_page_avail.insert(current_page * PAGE_SIZE);
//   }
//   GPU_phys_page_avail.reserve(GPU_total_memory_pages);
//   for (size_t current_page = 0; current_page < GPU_total_memory_pages; current_page++) {
//     GPU_phys_page_avail.insert(current_page * PAGE_SIZE);
//   }

//   for (Tensor *tensor : tensor_list) {
//     Addr starting_addr = tensor->address_offset;
//     long size_in_byte = (long) tensor->size_in_byte;
//     long total_pages = ceil((double) size_in_byte / PAGE_SIZE);
//     for (long page_num = 0; page_num < total_pages; page_num++) {
//       Addr page_starting_addr = starting_addr + PAGE_SIZE * page_num;
//       CPU_page_table[page_starting_addr] = CPUPageTableEntry();

//       CPUPageTableEntry &entry = CPU_page_table[page_starting_addr];
//       entry.ppn = 0;
//       if (is_UVM) {
//         if (tensor->is_global_weight) {
//           entry.location = IN_SSD; // UVM enabled, tensor started in SSD
//         } else {
//           entry.location = NOT_PRESENT; // ideal case, all tensor in GPU
//         }
//       } else {
//         entry.location = IN_GPU; // ideal case, all tensor in GPU
//       }
//       entry.in_transfer = PRESENT;
//     }
//   }

//   CPU_total_memory_pages = 0;

//   GPU_PCIe_remaining_cycles = 0;
//   PCIe_GPU_in_last_taken = IN_SSD;
//   PCIe_GPU_out_last_taken = IN_SSD;

//   GPU_malloc_remaining_cycles = 0;
//   GPU_free_remaining_cycles = 0;

//   current_cycle = 0;
//   reportGPUPageTable();
// }

// void Simulation::simulate() {
//   // globally used variables
//   vector<Tensor *> required_tensors;
//   printf("Start Simulation\n");

//   // simulation
//   int current_hint_ID = 0;
//   for (CUDAKernel kernel : kernel_list) {
//     required_tensors.clear();

//     int current_kernel_ID = kernel.kernel_id;
//     // consuming hints
//     while (current_hint_ID < movement_hints.size() &&
//         movement_hints[current_hint_ID].issued_time == current_kernel_ID) {
//       DataMovementHint &hint = movement_hints[current_hint_ID];
//       if (hint.from == NOT_PRESENT) {
//         // allocate
//         // Assert(hint.to != NOT_PRESENT);

//       } else if (hint.from == IN_GPU) {
//         // evict
//         // Assert(hint.to != IN_GPU);

//       } else {
//         // prefetch

//       }
//       current_hint_ID++;
//     }
//     unsigned long pervious_cycle = current_cycle;
//     printf("Kernel%d: <%s> @ %ld ================================\n",
//         current_kernel_ID,
//         print_kerneltype_array[kernel.type].c_str(),
//         current_cycle);

//     // get required tensors for current CUDA kernel
//     kernel.getRequiredTensors(required_tensors);

//     reportTensorStatus(required_tensors);
//     reportAllQueueStatus();

//     if (!requiredPageArrived(required_tensors)) {
//       iprintf("GPU start waiting critical page @ %ld\n", current_cycle);
//       do {
//         // if not all tensors have arrived transfer/alloc them in critical state
//         PageFaultInfo page_fault_info;
//         for (Tensor * tensor : required_tensors)
//           page_fault_info += transferPagesForTensor(tensor, true);
//         page_fault_info.kernel = &kernel;

//         // insert modeling black box here
//         unsigned long page_fault_end_cycle = getPageFaultTime(page_fault_info) + current_cycle;

//         unsigned long wait_cycles = GPUGetWaitingCriticalStepCycles();
//         iprintf("  GPU start waiting critical page @ %ld\n", current_cycle);
//         while (wait_cycles > 0) {
//           processGPUPCIeQueues(wait_cycles);
//           processGPUMemoryReqQueues(wait_cycles);
//           current_cycle += wait_cycles;
//           wait_cycles = GPUGetWaitingCriticalStepCycles();
//         }
//         wait_cycles = getGPUCriticalCompleteDeltaCycle();
//         processGPUPCIeQueues(wait_cycles);
//         processGPUMemoryReqQueues(wait_cycles);
//         current_cycle += wait_cycles;
//         iprintf("  GPU stop page fault handling @ %ld\n", current_cycle);

//         // Assert(page_fault_end_cycle > current_cycle);
//         current_cycle = page_fault_end_cycle;
//         iprintf("  Magic box end page fault handling @ %ld\n", current_cycle);
//         reportTensorStatus(required_tensors);
//         reportAllQueueStatus();
//       } while (!requiredPageArrived(required_tensors));
//       iprintf("GPU stop waiting critical page @ %ld\n", current_cycle);
//     }
//     // sanity check, if all the required data have arrived in the GPU memory
//     reportTensorStatus(required_tensors);
//     reportAllQueueStatus();
//     // Assert(requiredPageArrived(required_tensors));
//     // Assert(GPUGetWaitingCriticalStepCycles() == 0);

//     // runs as normal
//     unsigned long delta_cycles = kernel.execution_cycles;
//     processGPUPCIeQueues(delta_cycles);
//     processGPUMemoryReqQueues(delta_cycles);
//     current_cycle += delta_cycles;

//     // advance simulation to next kernel
//     reportGPUPageTable();
//     printf("Kernel%d <%ld> Ends @ Cycle %ld\n\n", kernel.kernel_id, current_cycle - pervious_cycle,
//         current_cycle);
//   }
//   printf("Simulation Ends @ Cycle %ld\n", current_cycle);
// }

// void Simulation::reportGPUPageTable() {
//   printf("GPU Page Table Available/Total=<%ld/%ld>\n",
//       GPU_phys_page_avail.size(), GPU_page_table.size() + GPU_phys_page_avail.size());
// }

// void Simulation::alloc_CPU_PTE(Addr vpn) {
//   // Assert(CPU_page_table.find(vpn) != CPU_page_table.end());
//   if (CPU_phys_page_avail.size() == 0) {
//     CPU_phys_page_avail.insert(CPU_total_memory_pages * PAGE_SIZE);
//     CPU_total_memory_pages++;
//   }
//   Addr ppn = *CPU_phys_page_avail.begin();
//   CPU_phys_page_avail.erase(ppn);
//   CPUPageTableEntry &entry = CPU_page_table[vpn];
//   entry.ppn = ppn;
//   entry.location = IN_CPU;
//   entry.in_transfer = IN_MIGRATION;
// }

// void Simulation::mark_invalid_CPU_PTE(Addr vpn) {
//   // Assert(CPU_page_table.find(vpn) != CPU_page_table.end());
//   CPUPageTableEntry &entry = CPU_page_table[vpn];
//   entry.in_transfer = IN_MIGRATION;
// }

// void Simulation::erase_CPU_PTE(Addr vpn) {
//   // Assert(CPU_page_table.find(vpn) != CPU_page_table.end());
//   CPUPageTableEntry &entry = CPU_page_table[vpn];
//   // Assert(CPU_phys_page_avail.find(vpn) == CPU_phys_page_avail.end());
//   CPU_phys_page_avail.insert(entry.ppn);
// }

// bool Simulation::alloc_GPU_PTE(Addr vpn) {
//   if (GPU_page_table.find(vpn) != GPU_page_table.end())
//     return true;
//   if (GPU_phys_page_avail.size() == 0) {
//     evict_GPU_PTE();
//     return false;
//   }
//   // Assert(GPU_phys_page_avail.size() > 0);
//   Addr ppn = *GPU_phys_page_avail.begin();
//   GPU_phys_page_avail.erase(ppn);
//   GPUPageTableEntry &entry = GPU_page_table[vpn];
//   entry.ppn = ppn;
//   entry.in_transfer = IN_MIGRATION;
//   return true;
// }

// void Simulation::free_GPU_PTE(Addr vpn) {
//   if (GPU_page_table.find(vpn) == GPU_page_table.end())
//     return;
//   GPUPageTableEntry &entry = GPU_page_table[vpn];
//   Addr ppn = entry.ppn;
//   // Assert(entry.in_transfer == PRESENT);
//   GPU_phys_page_avail.insert(ppn);
// }

// void Simulation::mark_invalid_GPU_PTE(Addr vpn) {
//   if (GPU_page_table.find(vpn) == GPU_page_table.end())
//     return;
//   GPUPageTableEntry &entry = GPU_page_table[vpn];
//   entry.in_transfer = IN_MIGRATION;
// }

// void Simulation::erase_GPU_PTE(Addr vpn) {
//   if (GPU_page_table.find(vpn) == GPU_page_table.end())
//     return;
//   GPU_page_table.erase(vpn);
//   GPU_phys_page_avail.insert(vpn);
// }

// Addr Simulation::evict_GPU_PTE() {
//   // GPU memory is full, eviction required
//   pair<Addr, GPUPageTableEntry> evicted_entry;
//   switch (evc_policy) {
//     case RANDOM: {
//       evicted_entry = *GPU_page_table.begin();
//       PageLocation dest = (rand() & 1) ? IN_SSD : IN_CPU;
//       if (rand() % 2) {
//         // send to SSD
//         GPU_to_SSD_PCIe_queue.addToFront(evicted_entry.first);
//       } else {
//         // send to CPU
//         GPU_to_CPU_PCIe_queue.addToFront(evicted_entry.first);
//       }
//       break;
//     }
//     case LRU: {
//       // Assert(false);
//       break;
//     }
//     case GUIDED: {
//       // Assert(false);
//       break;
//     }
//     default:
//       // Assert(false);
//   }
//   // Assert(GPU_page_table.find(evicted_entry.first) != GPU_page_table.end());
//   // Assert(CPU_page_table.find(evicted_entry.first) != CPU_page_table.end());
//   return evicted_entry.first;
// }

// void Simulation::PCIeQueue::reportQueueStatus() {
//   printf("  Queue %15s: inTransfer: %8ld, waiting: %8ld, critical: %8ld, total: %10ld\n",
//       name.c_str(), in_transfer_queue.size(), waiting_queue.size(), critical_pages.size(),
//       transferred_pages_history);
// }

// void Simulation::GPUMMU::reportMMUStatus() {
//   printf("  Queue %15s: inTransfer: %8ld, waiting: %8ld, critical: %8ld, total: %10ld\n",
//       name.c_str(), managing_queue.size(), waiting_queue.size(), critical_pages.size(),
//       transferred_pages_history);
// }

// bool Simulation::processCompleteEntryInGPUQueue(PCIeQueue &queue, unsigned long current_time) {
//   while (queue.hasTransferCompletePage(current_time)) {
//     // PCIe incoming movement completes
//     Addr completed_page = queue.getFrontCompletePage(current_time);
//     if (alloc_GPU_PTE(completed_page)) {
//       // successfully alloced
//       queue.removeFrontCompletePage(current_time);
//       GPU_page_table[completed_page].in_transfer = PRESENT;
//       // sync with CPU page table
//       CPU_page_table[completed_page].in_transfer = PRESENT;
//       CPU_page_table[completed_page].location = IN_GPU;
//     } else {
//       // alloc failed, wait for PTEs in GPU to be freed
//       return false;
//     }
//   }
//   return true;
// }

// void Simulation::processCompleteEntryInCPUQueue(unsigned long current_time) {
//   while (GPU_to_CPU_PCIe_queue.hasTransferCompletePage(current_time)) {
//     wprintf("test-1\n", "");
//     // PCIe incoming movement completes
//     Addr completed_page = GPU_to_CPU_PCIe_queue.getFrontCompletePage(current_time);
//     alloc_CPU_PTE(completed_page);
//     GPU_to_CPU_PCIe_queue.removeFrontCompletePage(current_time);
//     CPU_page_table[completed_page].in_transfer = PRESENT;
//     CPU_page_table[completed_page].location = IN_CPU;
//     // sync with GPU page table
//     if (GPU_page_table.find(completed_page) != GPU_page_table.end()) {
//       // entry not erased yet
//       // Assert(GPU_page_table[completed_page].in_transfer = IN_MIGRATION);
//       erase_GPU_PTE(completed_page);
//     }
//   }
// }

// void Simulation::processCompleteEntryInSSDQueue(unsigned long current_time) {
//   while (GPU_to_SSD_PCIe_queue.hasTransferCompletePage(current_time)) {
//     wprintf("test-2\n", "");
//     // PCIe incoming movement completes
//     Addr completed_page = GPU_to_SSD_PCIe_queue.getFrontCompletePage(current_time);
//     GPU_to_SSD_PCIe_queue.removeFrontCompletePage(current_time);
//     CPU_page_table[completed_page].in_transfer = PRESENT;
//     CPU_page_table[completed_page].location = IN_SSD;
//     // sync with GPU page table
//     if (GPU_page_table.find(completed_page) != GPU_page_table.end()) {
//       // entry not erased yet
//       // Assert(GPU_page_table[completed_page].in_transfer = IN_MIGRATION);
//       erase_GPU_PTE(completed_page);
//     }
//   }
// }

// void Simulation::processGPUPCIeQueues(unsigned long delta_cycles) {
//   // calculate how may batches can be created during the past <delta_cycles>
//   const unsigned long total_cycles = delta_cycles + GPU_PCIe_remaining_cycles;
//   const unsigned batching_num = total_cycles / PCIe_batch_ii_cycle;
//   // // GPU batcher running for <batching_num> times
//   //   wprintf("test-11 %lu %u\n", delta_cycles, batching_num);
//   for (unsigned batching_index = 0; batching_index < batching_num; batching_index++) {
//     unsigned long timestamp = current_cycle + batching_index * PCIe_batch_ii_cycle -
//         GPU_PCIe_remaining_cycles;
//     // PCIe events
//     GPUConsumeIncomingMovement(timestamp);
//     GPUHandleOutgoingMovement(timestamp);
//   }
//   processCompleteEntryInGPUQueue(CPU_to_GPU_PCIe_queue, current_cycle + delta_cycles);
//   processCompleteEntryInGPUQueue(SSD_to_GPU_PCIe_queue, current_cycle + delta_cycles);
//   processCompleteEntryInCPUQueue(current_cycle + delta_cycles);
//   processCompleteEntryInSSDQueue(current_cycle + delta_cycles);
//   GPU_PCIe_remaining_cycles = total_cycles - batching_num * PCIe_batch_ii_cycle;
// }

// void Simulation::processGPUMemoryReqQueues(unsigned long delta_cycles) {
//   // calculate how may page free can be handled during the past <delta_cycles>
//   const unsigned long GPU_free_total_cycles = delta_cycles + GPU_free_remaining_cycles;
//   const unsigned GPU_free_operation = GPU_free_total_cycles / GPU_free_cycle_per_page;
//   // GPU batcher running for <GPU_free_operation> times
//   for (unsigned operation_index = 0; operation_index < GPU_free_operation; operation_index++) {
//     unsigned long timestamp = current_cycle + operation_index * GPU_free_cycle_per_page -
//         GPU_free_remaining_cycles;
//     // free events
//     if (GPU_free_queue.needTransfer()) {
//       Addr freed_page = GPU_free_queue.transferFrontPage(timestamp);
//       erase_GPU_PTE(freed_page);
//     }
//   }
//   processCompleteEntryInGPUQueue(GPU_free_queue, current_cycle + delta_cycles);
//   GPU_free_remaining_cycles = GPU_free_total_cycles - GPU_free_operation * GPU_free_cycle_per_page;

//   // calculate how may page malloc can be handled during the past <delta_cycles>
//   const unsigned long GPU_malloc_total_cycles = delta_cycles + GPU_malloc_remaining_cycles;
//   const unsigned GPU_malloc_operation = GPU_malloc_total_cycles / GPU_malloc_cycle_per_page;
//   // GPU batcher running for <GPU_malloc_operation> times
//   for (unsigned operation_index = 0; operation_index < GPU_malloc_operation; operation_index++) {
//     unsigned long timestamp = current_cycle + operation_index * GPU_malloc_cycle_per_page -
//         GPU_malloc_remaining_cycles;
//     // malloc events
//     if (GPU_malloc_queue.needTransfer()) {
//       GPU_malloc_queue.transferFrontPage(timestamp);
//     }
//   }
//   processCompleteEntryInGPUQueue(GPU_malloc_queue, current_cycle + delta_cycles);
//   GPU_malloc_remaining_cycles = GPU_malloc_total_cycles - GPU_malloc_operation * GPU_malloc_cycle_per_page;
// }

// unsigned long Simulation::GPUGetWaitingCriticalStepCycles() {
//   vector<unsigned long> candidates;
//   if (SSD_to_GPU_PCIe_queue.hasWaitingCriticalPage())
//     candidates.push_back(SSD_to_GPU_PCIe_queue.latency_cycle);
//   if (GPU_to_SSD_PCIe_queue.hasWaitingCriticalPage())
//     candidates.push_back(GPU_to_SSD_PCIe_queue.latency_cycle);

//   if (CPU_to_GPU_PCIe_queue.hasWaitingCriticalPage())
//     candidates.push_back(CPU_to_GPU_PCIe_queue.latency_cycle);
//   if (GPU_to_CPU_PCIe_queue.hasWaitingCriticalPage())
//     candidates.push_back(GPU_to_CPU_PCIe_queue.latency_cycle);

//   if (GPU_malloc_queue.hasWaitingCriticalPage())
//     candidates.push_back(GPU_malloc_queue.latency_cycle);
//   if (GPU_free_queue.hasWaitingCriticalPage())
//     candidates.push_back(GPU_free_queue.latency_cycle);

//   unsigned long longest_latency_cycle = 0;
//   for (unsigned long candidate : candidates) {
//     longest_latency_cycle = longest_latency_cycle > candidate ?
//         longest_latency_cycle : candidate;
//   }
//   return longest_latency_cycle;
// }

// unsigned long Simulation::getGPUCriticalCompleteDeltaCycle() {
//   unsigned long abs_cycles[] = {
//     GPU_malloc_queue.getCriticalCompleteTime(),
//     CPU_to_GPU_PCIe_queue.getCriticalCompleteTime(),
//     SSD_to_GPU_PCIe_queue.getCriticalCompleteTime()
//   };
//   unsigned long to_return = 0;
//   for (unsigned long abs_cycle : abs_cycles)
//     to_return = to_return < abs_cycle ? abs_cycle : to_return;
//   // iprintf("<%lu,%lu,%lu>=>%lu | %lu\n", abs_cycles[0], abs_cycles[1], abs_cycles[2],
//   //     to_return, current_cycle);
//   if (to_return == 0 || to_return < current_cycle)
//     return 0;
//   return to_return - current_cycle;
// }

// void Simulation::GPUConsumeIncomingMovement(unsigned long timestamp) {
//   unsigned packed_num_page = 0;
//   while (packed_num_page < PCIe_batch_size_page) {
//     if (!CPU_to_GPU_PCIe_queue.needTransfer() && !SSD_to_GPU_PCIe_queue.needTransfer()) {
//       break;
//     }
//     // put entries from wait queue to in transfer queue
//     if (PCIe_GPU_in_last_taken == IN_SSD) {
//       if (CPU_to_GPU_PCIe_queue.needTransfer()) {
//         Addr transferred_page = CPU_to_GPU_PCIe_queue.transferFrontPage(timestamp);
//         mark_invalid_CPU_PTE(transferred_page);
//         packed_num_page++;
//       }
//       PCIe_GPU_in_last_taken = IN_CPU;
//     } else if (PCIe_GPU_in_last_taken == IN_CPU) {
//       if (SSD_to_GPU_PCIe_queue.needTransfer()) {
//         Addr transferred_page = SSD_to_GPU_PCIe_queue.transferFrontPage(timestamp);
//         mark_invalid_GPU_PTE(transferred_page);
//         packed_num_page++;
//       }
//       PCIe_GPU_in_last_taken = IN_SSD;
//     } else {
//       // Assert(false);
//     }
//   }
// }

// void Simulation::GPUHandleOutgoingMovement(unsigned long timestamp) {
//   unsigned packed_num_page = 0;
//   while (packed_num_page < PCIe_batch_size_page) {
//     if (!GPU_to_CPU_PCIe_queue.needTransfer() && !GPU_to_SSD_PCIe_queue.needTransfer()) {
//       break;
//     }
//     if (PCIe_GPU_out_last_taken == IN_SSD) {
//       if (GPU_to_CPU_PCIe_queue.needTransfer()) {
//         Addr transferred_page = GPU_to_CPU_PCIe_queue.transferFrontPage(timestamp);
//         mark_invalid_GPU_PTE(transferred_page);
//         packed_num_page++;
//       }
//       PCIe_GPU_out_last_taken = IN_CPU;
//     } else if (PCIe_GPU_out_last_taken == IN_CPU) {
//       if (GPU_to_SSD_PCIe_queue.needTransfer()) {
//         Addr transferred_page = GPU_to_SSD_PCIe_queue.transferFrontPage(timestamp);
//         mark_invalid_GPU_PTE(transferred_page);
//         packed_num_page++;
//       }
//       PCIe_GPU_out_last_taken = IN_SSD;
//     } else {
//       // Assert(false);
//     }
//   }
// }

// Simulation::PageFaultInfo Simulation::transferPagesForTensor(Tensor *tensor, bool is_critical) {
//   // Get pages that does not resides on the CPU
//   PageFaultInfo info;

//   Addr starting_addr = tensor->address_offset;
//   long size = (long) tensor->size_in_byte;
//   long total_pages = ceil((double) size / PAGE_SIZE);
//   for (long page_num = 0; page_num < total_pages; page_num++) {
//     Addr page_starting_addr = starting_addr + PAGE_SIZE * page_num;
//     CPUPageTableEntry &entry = CPU_page_table[page_starting_addr];
//     PCIeQueue *to_add_queue = nullptr;
//     if (entry.location == NOT_PRESENT) {
//       // add these pages to GPU malloc queue
//       info.not_presented_pages++;
//       to_add_queue = &GPU_malloc_queue;
//     } else if (entry.location == IN_SSD) {
//       // add these pages to SSD to GPU migration queue
//       info.SSD_to_GPU_faulted_pages++;
//       to_add_queue = &SSD_to_GPU_PCIe_queue;
//     } else if (entry.location == IN_CPU) {
//       // add these pages to CPU to GPU migration queue
//       info.CPU_to_GPU_faulted_pages++;
//       to_add_queue = &CPU_to_GPU_PCIe_queue;
//     } else {
//       continue;
//     }
//     if (is_critical)
//       to_add_queue->addToFront(page_starting_addr);
//     else
//       to_add_queue->addToBack(page_starting_addr);
//   }
//   return info;
// }

// unsigned long Simulation::getPageFaultTime(PageFaultInfo &info) {
//   // This should be replaced by a model in the future
//   const long CUDAMallocTimePerPageCycle   = 100000;
//   const long SSDPageFaultTimePerPageCycle = 240000;
//   const long CPUPageFaultTimePerPageCycle = 24000;

//   long total_cuda_malloc_pages = info.not_presented_pages +
//       info.SSD_to_GPU_faulted_pages + info.CPU_to_GPU_faulted_pages;
//   return info.SSD_to_GPU_faulted_pages * SSDPageFaultTimePerPageCycle +
//          info.CPU_to_GPU_faulted_pages * CPUPageFaultTimePerPageCycle +
//          total_cuda_malloc_pages * CUDAMallocTimePerPageCycle;
// }

// bool Simulation::requiredPageArrived(vector<Tensor *> &required_tensors) {
//   for (Tensor *tensor : required_tensors) {
//     Addr starting_addr = tensor->address_offset;
//     long size = (long) tensor->size_in_byte;
//     long total_pages = ceil((double) size / PAGE_SIZE);
//     for (long page_num = 0; page_num < total_pages; page_num++) {
//       Addr page_starting_addr = starting_addr + PAGE_SIZE * page_num;
//       CPUPageTableEntry &entry = CPU_page_table[page_starting_addr];
//       if (entry.location != IN_GPU)
//         return false;
//     }
//   }
//   return true;
// }

// void Simulation::PCIeQueue::addToFront(Addr page_start) {
//   // Assert(isPageAligned(page_start));
//   // already in transfer
//   if (metadata.find(page_start) != metadata.end() && metadata[page_start] >= 0)
//     return;
//   // if found in wait queue, requeue at start of the queue
//   // Assert(metadata.find(page_start) == metadata.end() || metadata[page_start] < 0);
//   if (metadata.find(page_start) != metadata.end() && metadata[page_start] < 0) {
//     waiting_queue.erase(std::remove(waiting_queue.begin(), waiting_queue.end(), page_start), waiting_queue.end());
//     metadata.erase(page_start);
//   }
//   waiting_queue.push_front(page_start);
//   metadata[page_start] = -1;
//   critical_pages.insert(page_start);
// }

// void Simulation::PCIeQueue::addToBack(Addr page_start) {
//   // Assert(isPageAligned(page_start));
//   // if found in wait queue, do nothing
//   if (metadata.find(page_start) != metadata.end())
//     return;
//   waiting_queue.push_back(page_start);
//   metadata[page_start] = -1;
// }

// bool Simulation::PCIeQueue::needTransfer() {
//   return waiting_queue.size() > 0;
// }

// Addr Simulation::PCIeQueue::transferFrontPage(unsigned long timestamp) {
//   // Assert(waiting_queue.size() > 0);
//   Addr to_transfer = *waiting_queue.begin();
//   if (metadata.find(to_transfer) != metadata.end() && metadata[to_transfer] != -1) {
//     eprintf("@ %lu =>%08lX\n", timestamp, to_transfer);
//     reportQueueStatus();
//     print();
//   }
//   // Assert(metadata.find(to_transfer) != metadata.end() && metadata[to_transfer] == -1);
//   metadata[to_transfer] = timestamp + latency_cycle;
//   waiting_queue.erase(waiting_queue.cbegin());
//   in_transfer_queue.push_back(to_transfer);
//   return to_transfer;
// }

// bool Simulation::PCIeQueue::hasTransferCompletePage(unsigned long timestamp) {
//   if (in_transfer_queue.size() == 0)
//     return false;
//   // Assert(metadata.find(waiting_queue.front()) != metadata.end());
//   long front_entry_timestamp = metadata[in_transfer_queue.front()];
//   // Assert(front_entry_timestamp >= 0);
//   return timestamp >= front_entry_timestamp;
// }

// bool Simulation::PCIeQueue::hasTransferringCriticalPage() {
//   return critical_pages.size() != 0;
// }

// bool Simulation::PCIeQueue::hasWaitingCriticalPage() {
//   if (waiting_queue.size() == 0)
//     return false;
//   return critical_pages.find(waiting_queue.front()) != critical_pages.end();
// }

// Addr Simulation::PCIeQueue::getFrontCompletePage(unsigned long timestamp) {
//   // Assert(hasTransferCompletePage(timestamp));
//   return in_transfer_queue.front();
// }

// unsigned long Simulation::PCIeQueue::getCriticalCompleteTime() {
//   // Assert(!hasWaitingCriticalPage());
//   unsigned long abs_cycle = 0;
//   for (Addr page : critical_pages) {
//     // Assert(metadata.find(page) != metadata.end() && metadata[page] != -1);
//     abs_cycle = abs_cycle < metadata[page] ? metadata[page] : abs_cycle;
//   }
//   return abs_cycle;
// }

// void Simulation::PCIeQueue::removeFrontCompletePage(unsigned long timestamp) {
//   // Assert(hasTransferCompletePage(timestamp));
//   Addr completed_page = in_transfer_queue.front();
//   if (critical_pages.find(completed_page) != critical_pages.end())
//     critical_pages.erase(completed_page);
//   metadata.erase(completed_page);
//   in_transfer_queue.pop_front();
//   transferred_pages_history++;
// }

// void Simulation::GPUMMU::addToQueue(Addr page_start, unsigned long size,
//                                     MMUAction action, bool critical) {
//   // Assert(isPageAligned(page_start));
//   // Assert(isPageSized(size));
//   // create a new entry
//   GPUMMUEntry new_entry;
//   new_entry.page_start = page_start;
//   new_entry.size = size;
//   new_entry.action = action;
//   new_entry.is_critical = critical;
//   if (critical) {
//     waiting_queue.push_front(new_entry);
//     critical_pages_count++;
//   } else {
//     waiting_queue.push_back(new_entry);
//   }

//   if (action == ALLOC)
//     alloc_request_num++;
//   else if (action == FREE)
//     free_request_num++;
//   else
//     // Assert(false);
// }

// bool Simulation::GPUMMU::needAction() {
//   return waiting_queue.size() > 0;
// }

// Simulation::GPUMMU::GPUMMUEntry Simulation::GPUMMU::manageFrontRequest(unsigned long timestamp) {
//   // Assert(waiting_queue.size() > 0);
//   GPUMMUEntry managing_entry = waiting_queue.front();
//   if (managing_entry.action == ALLOC) {
//     managing_entry.finishing_timestamp = timestamp +
//         alloc_lat_off + alloc_lat_slp * managing_entry.size;
//   } else if (managing_entry.action == FREE) {
//     managing_entry.finishing_timestamp = timestamp +
//         free_lat_off + free_lat_slp * managing_entry.size;
//   } else {
//     // Assert(false);
//   }
//   managing_queue.push_back(managing_entry);
//   waiting_queue.erase(waiting_queue.cbegin());
//   return managing_entry;
// }

// bool Simulation::GPUMMU::hasCompletedRequest(unsigned long timestamp) {
//   if (managing_queue.size() == 0)
//     return false;
//   long front_entry_timestamp = managing_queue.front().finishing_timestamp;
//   return timestamp >= front_entry_timestamp;
// }

// bool Simulation::GPUMMU::hasManagingCriticalRequest() {
//   return critical_pages_count != 0;
// }

// bool Simulation::GPUMMU::hasWaitingCriticalRequest() {
//   if (waiting_queue.size() == 0)
//     return false;
//   GPUMMUEntry &front_entry = waiting_queue.front();
//   return front_entry.is_critical;
// }

// bool Simulation::GPUMMU::getFrontCompletedRequest(unsigned long timestamp,
//     Simulation::GPUMMU::GPUMMUEntry &entry) {
//   if (!hasCompletedRequest(timestamp))
//     return false;
//   entry = managing_queue.front();
//   return true;
// }

// bool Simulation::GPUMMU::removeFrontCompletedRequest(unsigned long timestamp) {
//   if (!hasCompletedRequest(timestamp))
//     return false;
//   GPUMMUEntry &entry = managing_queue.front();
//   if (entry.action == ALLOC)
//     alloc_request_num++;
//   else if (entry.action == FREE)
//     free_request_num++;
//   else
//     // Assert(false);
//   managing_queue.pop_front();
//   return true;
// }

// void Simulation::reportTensorStatus(vector<Tensor *> &tensors) {
//   printf("  num: %ld\n", tensors.size());
//   for (Tensor *tensor : tensors) {
//     Addr starting_addr = tensor->address_offset;
//     long size = (long) tensor->size_in_byte;
//     long total_pages = ceil((double) size / PAGE_SIZE);

//     long in_migration_from_cpu_pages = 0;
//     long in_migration_from_ssd_pages = 0;
//     long in_malloc_pages = 0;
//     long in_migration_out_pages = 0;
//     long miss_located_cpu_pages = 0;
//     long miss_located_ssd_pages = 0;
//     long need_malloc_pages = 0;

//     long total_in_migration_pages = 0;
//     long total_miss_located_pages = 0;
//     long total_in_place_pages = 0;
//     for (long page_num = 0; page_num < total_pages; page_num++) {
//       if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].in_transfer == IN_MIGRATION) {
//         if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].location == IN_CPU) {
//           in_migration_from_cpu_pages++;
//         } else if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].location == IN_SSD) {
//           in_migration_from_ssd_pages++;
//         } else if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].location == NOT_PRESENT) {
//           in_malloc_pages++;
//         } else if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].location == IN_GPU) {
//           in_migration_out_pages++;
//         } else {
//           // Assert(false);
//         }
//       } else if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].in_transfer == PRESENT) {
//         if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].location == IN_CPU) {
//           miss_located_cpu_pages++;
//         } else if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].location == IN_SSD) {
//           miss_located_ssd_pages++;
//         } else if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].location == NOT_PRESENT) {
//           need_malloc_pages++;
//         } else if (CPU_page_table[starting_addr + PAGE_SIZE * page_num].location == IN_GPU) {
//           total_in_place_pages++;
//         } else {
//           // Assert(false);
//         }
//       } else {
//         // Assert(false);
//       }
//     }
//     total_in_migration_pages = in_migration_from_cpu_pages + in_migration_from_ssd_pages +
//         in_malloc_pages + in_migration_out_pages;
//     total_miss_located_pages = miss_located_cpu_pages + miss_located_ssd_pages + need_malloc_pages;
//     if (total_in_migration_pages > 0 || total_miss_located_pages > 0) {
//       printf("  Tensor <%s> <IN_MIGRATION:%-5ld=CPU:%-5ld+SSD:%-5ld+MALLOC:%-5ld+EVICT:%-5ld>"
//           "+<MISS_LOCATED:%-5ld=CPU:%-5ld+SSD:%-5ld+MALLOC:%-5ld>+<IN_PLACE:%-5ld>"
//           "/<TOTAL:%-5ld> not in GPU\n",
//           tensor->name.c_str(),
//           total_in_migration_pages, in_migration_from_cpu_pages, in_migration_from_ssd_pages,
//           in_malloc_pages, in_migration_out_pages,
//           total_miss_located_pages, miss_located_cpu_pages, miss_located_ssd_pages,
//           need_malloc_pages,
//           total_in_place_pages, total_pages);
//     } else {
//       printf("  Tensor <%s> is in GPU\n", tensor->name.c_str());
//     }
//   }
// }

// void Simulation::PCIeQueue::print() {
//   if (waiting_queue.size() > 0) {
//     printf("Wait Queue:\n");
//     for (Addr start_addr : waiting_queue) {
//       printf("<[%010lX-%010lX], %18ld> <= ", start_addr, start_addr + PAGE_SIZE, metadata[start_addr]);
//     }
//     printf("\n");
//   }
//   if (in_transfer_queue.size() > 0) {
//     printf("In Transfer Queue:\n");
//     for (Addr start_addr : in_transfer_queue) {
//       printf("<[%010lX-%010lX], %18ld> <= ", start_addr, start_addr + PAGE_SIZE, metadata[start_addr]);
//     }
//     printf("\n");
//   }
// }

// void Simulation::reportAllQueueStatus() {
//   GPU_malloc_queue.reportQueueStatus();
//   GPU_free_queue.reportQueueStatus();
//   CPU_to_GPU_PCIe_queue.reportQueueStatus();
//   SSD_to_GPU_PCIe_queue.reportQueueStatus();
//   GPU_to_CPU_PCIe_queue.reportQueueStatus();
//   GPU_to_SSD_PCIe_queue.reportQueueStatus();
// }
