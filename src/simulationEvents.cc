#include <math.h>
#include <unistd.h>

#include "simulationEvents.h"
#include "simulationComponents.h"
#include "printUtils.h"
#include "analysis.h"

using std::get;
using std::max;
using std::ceil;
using std::pair;
using std::string;
using std::make_pair;
using std::make_tuple;
using Simulator::DataMovementHint;
using Simulator::PageLocation;

extern vector<Tensor*> tensor_list;
extern vector<CUDAKernel> kernel_list;
extern vector<Model_Layer*> forward_layers;
extern vector<DataMovementHint> movement_hints;

extern string output_folder_name;

namespace Simulator {

System *sim_sys;
Stat *sim_stat;

// KernelBeginEvent BEGIN ========================
bool KernelBeginEvent::requiredPageArrived(vector<Tensor *> &required_tensors, bool overtime) {
  if (sim_sys->getCurrentTotalPF() > 0) return false;
  // should not print if overtime
  bool arrived = true;
  for (Tensor *tensor : required_tensors) {
    Addr start_addr = tensor->getGlobalOffset();
    long size = (long) tensor->size_in_byte;
    long total_pages = ceil((double) size / PAGE_SIZE);

    long in_migration_from_cpu_pages = 0;
    long in_migration_from_ssd_pages = 0;
    long in_malloc_pages = 0;
    long in_migration_out_pages = 0;
    long miss_located_cpu_pages = 0;
    long miss_located_ssd_pages = 0;
    long need_malloc_pages = 0;

    long total_in_migration_pages = 0;
    long total_miss_located_pages = 0;
    long total_in_place_pages = 0;
    CPUPageTable::CPUPageTableEntry *entry;
    for (long page_num = 0; page_num < total_pages; page_num++) {
      entry = sim_sys->CPU_PT.getEntry(start_addr + PAGE_SIZE * page_num);
      Assert(entry);
      if (overtime) {
        if (entry->in_transfer || entry->location != IN_GPU)
          return false;
      } else {
        if (entry->in_transfer || entry->location != IN_GPU)
          arrived = false;
        if (entry->in_transfer) {
          if (entry->location == IN_CPU) {
            in_migration_from_cpu_pages++;
          } else if (entry->location == IN_SSD) {
            in_migration_from_ssd_pages++;
          } else if (entry->location == NOT_PRESENT) {
            in_malloc_pages++;
          } else if (entry->location == IN_GPU) {
            in_migration_out_pages++;
          } else {
            Assert(false);
          }
        } else {
          if (entry->location == IN_CPU) {
            miss_located_cpu_pages++;
          } else if (entry->location == IN_SSD) {
            miss_located_ssd_pages++;
          } else if (entry->location == NOT_PRESENT) {
            need_malloc_pages++;
          } else if (entry->location == IN_GPU) {
            total_in_place_pages++;
          } else {
            Assert(false);
          }
        }
      }
    }
    if (!overtime) {
      total_in_migration_pages = in_migration_from_cpu_pages + in_migration_from_ssd_pages +
          in_malloc_pages + in_migration_out_pages;
      total_miss_located_pages = miss_located_cpu_pages + miss_located_ssd_pages + need_malloc_pages;
      if (total_in_migration_pages > 0 || total_miss_located_pages > 0) {
        printf("   ⊢%13s ", tensor->name().c_str());
        if (tensor->is_global_weight) printf("[  Global   ] ");
        else printf("[%5d-%5d] ", tensor->live_interval[0], tensor->live_interval[1]);
        printf("<IN_MIGRATION:%-8ld=CPU:%-8ld+SSD:%-8ld+MALLOC:%-8ld+EVICT:%-8ld>"
            "+<MISS_LOCATED:%-8ld=CPU:%-8ld+SSD:%-8ld+MALLOC:%-8ld>+<IN_PLACE:%-8ld>"
            "/<TOTAL:%-8ld>\n",
            total_in_migration_pages, in_migration_from_cpu_pages, in_migration_from_ssd_pages,
            in_malloc_pages, in_migration_out_pages,
            total_miss_located_pages, miss_located_cpu_pages, miss_located_ssd_pages,
            need_malloc_pages, total_in_place_pages, total_pages);
        sim_stat->addPFTensor(sim_sys->getCurrentIteration(), tensor, total_pages,
            in_migration_from_cpu_pages, in_migration_from_ssd_pages, in_malloc_pages,
            miss_located_cpu_pages, miss_located_ssd_pages, need_malloc_pages);
      } else {
        printf("   ⊢%13s ", tensor->name().c_str());
        if (tensor->is_global_weight) printf("[  Global   ] ");
        else printf("[%5d-%5d] ", tensor->live_interval[0], tensor->live_interval[1]);
        printf("is in GPU\n");
      }
    }
  }
  return arrived;
}

PageFaultInfo KernelBeginEvent::transferTensorToGPU(Tensor *tensor, bool is_input) {
  // Get pages that does not resides on the GPU
  PageFaultInfo info(nullptr);

  Addr start_addr = tensor->getGlobalOffset();
  long size = (long) tensor->size_in_byte;
  long total_pages = ceil((double) size / PAGE_SIZE);

  if (is_input) info.total_input_pages += total_pages;
  else info.total_output_pages += total_pages;

  PageLocation destination = PageLocation::NOT_PRESENT;
  for (long page_num = 0; page_num < total_pages; page_num++) {
    Addr page_starting_addr = start_addr + PAGE_SIZE * page_num;
    CPUPageTable::CPUPageTableEntry *CPU_PTE =
        sim_sys->CPU_PT.getEntry(page_starting_addr);
    Assert(CPU_PTE);
    if (CPU_PTE->in_transfer) {
      info.in_transfer_pages++;
      continue;
    }
    // critical operation
    if (CPU_PTE->location == NOT_PRESENT) {
      if (is_input) info.not_presented_input_pages++;
      else info.not_presented_output_pages++;
      sim_sys->pf_alloc_queue.push_back(page_starting_addr);
    } else if (CPU_PTE->location == IN_SSD) {
      if (is_input) info.SSD_to_GPU_faulted_input_pages++;
      else info.SSD_to_GPU_faulted_output_pages++;
      sim_sys->pf_SSD_queue.push_back(page_starting_addr);
      destination = PageLocation::IN_GPU;
    } else if (CPU_PTE->location == IN_CPU) {
      if (is_input) info.CPU_to_GPU_faulted_input_pages++;
      else info.CPU_to_GPU_faulted_output_pages++;
      sim_sys->pf_CPU_queue.push_back(page_starting_addr);
    } else if (CPU_PTE->location == IN_GPU) {
      continue;
    } else {
      Assert(false);
    }
    sim_sys->CPU_PT.markInTransferPTE(page_starting_addr);
  }
  if (destination == PageLocation::IN_GPU)
    sim_stat->addSizeInfo(tensor->raw_size_byte, tensor->size_in_byte);
  return info;
}

void KernelBeginEvent::guidedTransfer(DataMovementHint *hint) {
  Tensor *tensor = hint->tensor;
  Addr start_addr = tensor->getGlobalOffset();
  long size = (long) tensor->size_in_byte;
  long total_pages = ceil((double) size / PAGE_SIZE);
  for (long page_num = 0; page_num < total_pages; page_num++) {
    Addr page_starting_addr = start_addr + PAGE_SIZE * page_num;
    CPUPageTable::CPUPageTableEntry *CPU_PTE =
        sim_sys->CPU_PT.getEntry(page_starting_addr);
    Assert(CPU_PTE);
    if (CPU_PTE->in_transfer)
      continue;

    if (hint->from == PageLocation::IN_GPU_LEAST) {
      Assert(hint->to == PageLocation::IN_GPU);
      if (sim_sys->GPU_PT.exist(page_starting_addr)) {
        // Unpin
        sim_sys->GPU_PT.LRUUnpin(page_starting_addr);
      }
      hint->human_readable_hint = "Unpin";
    } else if (hint->to == PageLocation::IN_GPU_LEAST) {
      Assert(hint->from == PageLocation::IN_GPU);
      if (sim_sys->GPU_PT.exist(page_starting_addr)) {
        // Pin
        sim_sys->GPU_PT.LRUPin(page_starting_addr);
      }
      hint->human_readable_hint = "Pin";
    } else if (hint->to == PageLocation::IN_GPU) {
      if (CPU_PTE->location == PageLocation::NOT_PRESENT) {
        // prealloc
        sim_sys->prealloc_queue.push_back(page_starting_addr);
        hint->human_readable_hint = "Prealloc";
        sim_sys->CPU_PT.markInTransferPTE(page_starting_addr);
      } else if (CPU_PTE->location == PageLocation::IN_SSD) {
        // prefetch from SSD
        sim_sys->prefetch_SSD_queue.push_back(page_starting_addr);
        hint->human_readable_hint = "Prefetch";
        sim_sys->CPU_PT.markInTransferPTE(page_starting_addr);
      } else if (CPU_PTE->location == PageLocation::IN_CPU) {
        // prefetch from CPU
        sim_sys->prefetch_CPU_queue.push_back(page_starting_addr);
        hint->human_readable_hint = "Prefetch";
        sim_sys->CPU_PT.markInTransferPTE(page_starting_addr);
      }
    } else if (hint->to == PageLocation::IN_SSD || hint->to == PageLocation::IN_CPU) {
      if (CPU_PTE->location == PageLocation::IN_GPU) {
        // pre-evict
        if (hint->to == PageLocation::IN_SSD) {
          sim_sys->preevict_SSD_queue.push_back(page_starting_addr);
        } else {
          sim_sys->preevict_CPU_queue.push_back(page_starting_addr);
        }
        hint->human_readable_hint = "Pre-evict";
        sim_sys->CPU_PT.markInTransferPTE(page_starting_addr);
      }
    } else if (hint->to == PageLocation::NOT_PRESENT) {
      // pre-dealloc, immediate
      if (CPU_PTE->location == PageLocation::IN_GPU) {
        if (sim_sys->GPU_PT.exist(page_starting_addr))
          sim_sys->GPU_PT.erasePTE(page_starting_addr);
      } else if (CPU_PTE->location == PageLocation::IN_SSD) {
      } else if (CPU_PTE->location == PageLocation::IN_CPU) {
        if (sim_sys->CPU_PT.exist(page_starting_addr))
          sim_sys->CPU_PT.erasePTE(page_starting_addr);
      }
      CPU_PTE->location = PageLocation::NOT_PRESENT;
      hint->human_readable_hint = "Predealloc";
      Assert(!sim_sys->GPU_PT.exist(page_starting_addr));
    } else {
      hint->human_readable_hint = "Ignored";
    }
  }
}

unsigned long KernelBeginEvent::getPageFaultTime(PageFaultInfo &info) {
  double deltaT_PF, BW_ssd_rest, BW_pcie_rest;
  double input_pf_ratio = info.total_input_pages != 0 ? (double) 
      (info.CPU_to_GPU_faulted_input_pages + info.SSD_to_GPU_faulted_input_pages + info.not_presented_input_pages) / 
      info.total_input_pages :
      0;
  double output_pf_ratio = info.total_output_pages != 0 ? (double) 
      (info.CPU_to_GPU_faulted_output_pages + info.SSD_to_GPU_faulted_output_pages + info.not_presented_output_pages) / 
      info.total_output_pages :
      0;
  double input_pf_SSD_ratio = 
      info.CPU_to_GPU_faulted_input_pages + info.SSD_to_GPU_faulted_input_pages + info.not_presented_input_pages != 0 ?
      (double) info.SSD_to_GPU_faulted_input_pages / 
      (info.CPU_to_GPU_faulted_input_pages + info.SSD_to_GPU_faulted_input_pages + info.not_presented_input_pages) :
      0;
  double output_pf_SSD_ratio = 
      info.CPU_to_GPU_faulted_output_pages + info.SSD_to_GPU_faulted_output_pages + info.not_presented_output_pages != 0 ?
      (double) info.SSD_to_GPU_faulted_output_pages / 
      (info.CPU_to_GPU_faulted_output_pages + info.SSD_to_GPU_faulted_output_pages + info.not_presented_output_pages) :
      0;
  unsigned long input_tensor_size = info.total_input_pages * PAGE_SIZE;
  unsigned long output_tensor_size = info.total_output_pages * PAGE_SIZE;
  nprintf("  Kernel %d PF -> 00: %12lld 01: %12lld 11: %12lld InputSize: %18lld OutputSize: %18lld\n"
          "    CPU input PF %12lld SSD input PF %12lld CPU output PF %12lld SSD output PF %12lld\n"
          "    InputPFRatio: %7.3f OutputPFRatio: %7.3f InputPFSSDRatio: %7.3f OutputPFSSDRatio: %7.3f\n",
      info.kernel->kernel_id, 
      info.kernel->pf_execution_cycles, info.kernel->input_pf_execution_cycles, info.kernel->execution_cycles,
      input_tensor_size, output_tensor_size,
      info.CPU_to_GPU_faulted_input_pages, info.SSD_to_GPU_faulted_input_pages,
      info.CPU_to_GPU_faulted_output_pages, info.SSD_to_GPU_faulted_output_pages,
      input_pf_ratio, output_pf_ratio, input_pf_SSD_ratio, output_pf_SSD_ratio);
  performance_model(kernel->pf_execution_cycles / sim_sys->GPU_frequency_Hz * 1000, 
                    kernel->input_pf_execution_cycles / sim_sys->GPU_frequency_Hz * 1000, 
                    kernel->execution_cycles / sim_sys->GPU_frequency_Hz * 1000,
                    input_pf_ratio, output_pf_ratio, input_pf_SSD_ratio, output_pf_SSD_ratio,
                    input_tensor_size, output_tensor_size,
                    sim_sys->GPU_PCIe_bandwidth_Bpc * sim_sys->GPU_frequency_Hz / 1000, 
                    sim_sys->SSD_PCIe_bandwidth_Bpc * sim_sys->GPU_frequency_Hz / 1000,
                    (int) sim_sys->system_latency, (int) sim_sys->SSD_latency,
                    deltaT_PF, BW_ssd_rest, BW_pcie_rest);
  unsigned long delta_cycle = deltaT_PF / pow(10, 3) * sim_sys->GPU_frequency_Hz;
  nprintf("    Model predicts %f ms %lld cycles, total %lld cycles\n", 
      deltaT_PF, delta_cycle, delta_cycle + kernel->execution_cycles);
  return delta_cycle + kernel->execution_cycles;
}

bool KernelBeginEvent::shouldExecute() {
  if (kernel != sim_sys->getCurrentKernel())
    sim_sys->advanceCurrentKernel();
  Assert(kernel == sim_sys->getCurrentKernel());
  return sim_sys->getCurrentIteration() < sim_sys->getMaxIteration();
}

void KernelBeginEvent::execute(vector<Event *> &created_events) {
  // print facilities
  sim_sys->batcher_evt_print_current = 0;
  // change executing kernel if informed
  if (kernel != sim_sys->getCurrentKernel()) {
    sim_sys->advanceCurrentKernel();
  }
  sim_sys->generateRequiredTensorsForCurrentKernel();

  // generate DEEPUM specific LRU suggestion for each of the kernel
  if (sim_sys->mig_policy == MigPolicy::DEEPUM) {
    sim_sys->clearRunningWindow();
    for (int kernel_offset = 0; kernel_offset < sim_sys->sys_prefetch_degree; kernel_offset++) 
      sim_sys->addKernelTensorsToRunningWindow(
          (sim_sys->getCurrentKernel()->kernel_id + kernel_offset) % kernel_list.size());
    sim_sys->deepUMSuggestInitialLRUBase();
  }
  // generate specific LRU suggestion for each bacher event for each of LRU-based algo
  else if (sim_sys->evc_policy == GPUPageTable::EvcPolicy::LRU ||
      sim_sys->evc_policy == GPUPageTable::EvcPolicy::GUIDED_LRU || 
      sim_sys->evc_policy == GPUPageTable::EvcPolicy::PERFECT_GUIDED_LRU) {
    sim_sys->LRUSuggestInitialLRUBase();
  }
  
  // get required tensors for current CUDA kernel
  vector<Tensor *> required_tensors;
  vector<Tensor *> required_input_tensors;
  vector<Tensor *> required_output_tensors;
  kernel->getRequiredTensors(required_tensors, required_input_tensors, required_output_tensors);
  sim_sys->CPU_PT.report();
  sim_sys->GPU_PT.report();

  // prefetch, pre(de)alloc, pre-evict guide
  if (sim_sys->should_prefetch) {
    vector<DataMovementHint> current_hints;
    sim_sys->getCurrentMovementHints(current_hints);
    if (current_hints.size() != 0)
      iprintf("  Guide report\n", "");
    for (DataMovementHint hint : current_hints) {
      Assert(hint.issued_time % kernel_list.size() == sim_sys->getCurrentKernel()->kernel_id);
      guidedTransfer(&hint);
      printf("   ⊢%13s ", hint.tensor->name().c_str());
      if (hint.tensor->is_global_weight) printf("[  Global   ] ");
      else printf("[%5d-%5d] ", hint.tensor->live_interval[0], hint.tensor->live_interval[1]);
      printf("From: %11s, To: %11s, %s\n",
          print_pagelocation_array[hint.from].c_str(),
          print_pagelocation_array[hint.to].c_str(),
          hint.human_readable_hint.c_str());
    }
  }
  sim_sys->CPU_PT.report();
  sim_sys->GPU_PT.report();

  // page fault
  bool overtime = sim_sys->reschedule_info &&
                  (sim_sys->reschedule_info->first_scheduled_time +
                  sim_sys->reschedule_info->page_faulted_time <=
                  scheduled_time);
  if (sim_sys->reschedule_info) {
    printf("  First scheduled time: %ld, PF exe time: %ld, Max ending time: %ld, Current time: %ld, Overtime: %s\n", 
        sim_sys->reschedule_info->first_scheduled_time, sim_sys->reschedule_info->page_faulted_time,
        sim_sys->reschedule_info->first_scheduled_time + sim_sys->reschedule_info->page_faulted_time,
        scheduled_time, overtime ? "o" : "x");
  } else {
    printf("  First scheduled time: %ld, PF exe time: %ld, Max ending time: %ld, Current time: %ld\n", 
        scheduled_time, kernel->pf_execution_cycles,
        scheduled_time + kernel->pf_execution_cycles,
        scheduled_time);
  }
  if (!overtime)
    iprintf("  Tensor report\n", "");
  if (!requiredPageArrived(required_tensors, overtime)) {
    // if not all tensors required by the following kernel arrived
    nprintf("  Kernel%d req tensors not yet arrived\n", kernel->kernel_id);
    // get page fault info and start transfer required tensors
    PageFaultInfo page_fault_info(kernel);
    for (Tensor *tensor : required_input_tensors)
      page_fault_info += transferTensorToGPU(tensor, true);
    for (Tensor *tensor : required_output_tensors)
      page_fault_info += transferTensorToGPU(tensor, false);
    sim_sys->CPU_PT.AddInTransferPages(required_tensors);
    iprintf("  PF NUM: %ld [Alloc: %ld, SSD: %ld, CPU: %ld], IN_TRANSFER: %ld\n",
        sim_sys->pf_alloc_queue.size() + sim_sys->pf_SSD_queue.size() + sim_sys->pf_CPU_queue.size(),
        sim_sys->pf_alloc_queue.size(), sim_sys->pf_SSD_queue.size(), sim_sys->pf_CPU_queue.size(), 
        page_fault_info.in_transfer_pages);
    // mark this kernel needed to be processed when all page fault resolved
    unsigned long page_faulted_time = getPageFaultTime(page_fault_info);
    if (!sim_sys->reschedule_info) {
      sim_sys->reschedule_info =
          new KernelRescheduleInfo(scheduled_time, page_faulted_time);
      // LRU visualization ////////////////////////////////////////////////////////////
      // char filename[100];
      // snprintf(filename, sizeof(filename), "%s/lru_trace/LRUReport%d.%05d.report", 
      //     output_folder_name.c_str(), sim_sys->getCurrentIteration(), kernel->kernel_id);
      // ofstream fout(filename, ofstream::app);
      // fout << sim_sys->GPU_PT.reportLRUTable(kernel->kernel_id);
      // fout.close();
      /////////////////////////////////////////////////////////////////////////////////
    } else if (overtime) {
      wprintf("  Ignoring time for %ld CPU pages, %ld SSD pages, %ld un-alloc pages, %ld in transfer pages\n",
          page_fault_info.CPU_to_GPU_faulted_input_pages + page_fault_info.CPU_to_GPU_faulted_output_pages,
          page_fault_info.SSD_to_GPU_faulted_input_pages + page_fault_info.SSD_to_GPU_faulted_output_pages,
          page_fault_info.not_presented_input_pages + page_fault_info.not_presented_output_pages,
          page_fault_info.in_transfer_pages);
    }
    Assert(sim_sys->reschedule_info);
    Assert(!sim_sys->reschedule_info->kernel_started);
    sim_sys->reschedule_info->PF_info.push_back(page_fault_info);
    // page faults time brought by eviction is ignored
  } else {
    // if all tensors required by the following kernel arrived
    iprintf("  Kernel%d req tensors all arrived\n", kernel->kernel_id);
    unsigned long to_schedule_time;
    if (sim_sys->reschedule_info) {
      // rescheduled
      to_schedule_time = sim_sys->reschedule_info->first_scheduled_time +
                         sim_sys->reschedule_info->page_faulted_time;
      Assert(sim_sys->reschedule_info->kernel_started);

      if (to_schedule_time < scheduled_time) {
        wprintf("  Page fault time < rescheduling time\n", "");
        to_schedule_time = scheduled_time;
      }
      sim_stat->addKernelStat(sim_sys->getCurrentIteration(),
                              sim_sys->reschedule_info->first_scheduled_time,
                              to_schedule_time,
                              sim_sys->CPU_PT.getCapacity().first,
                              sim_sys->GPU_PT.getCapacity().first,
                              kernel,
                              sim_sys->reschedule_info->PF_info);

      delete sim_sys->reschedule_info;
      sim_sys->reschedule_info = nullptr;
    } else {
      // not rescheduled
      to_schedule_time = scheduled_time + kernel->execution_cycles;
      sim_stat->addKernelStat(sim_sys->getCurrentIteration(),
                              scheduled_time, to_schedule_time, 
                              sim_sys->CPU_PT.getCapacity().first,
                              sim_sys->GPU_PT.getCapacity().first,
                              kernel);
      // LRU visualization ////////////////////////////////////////////////////////////
      // char filename[100];
      // snprintf(filename, sizeof(filename), "%s/lru_trace/LRUReport%d.%05d.report", 
      //     output_folder_name.c_str(), sim_sys->getCurrentIteration(), kernel->kernel_id);
      // ofstream fout(filename, ofstream::app);
      // fout << sim_sys->GPU_PT.reportLRUTable(kernel->kernel_id);
      // fout.close();
      /////////////////////////////////////////////////////////////////////////////////
    }
    // scheduled new kernel start event
    Assert(!sim_sys->reschedule_info);
    created_events.push_back(
        new KernelBeginEvent(to_schedule_time, sim_sys->getNextKernel()));
    sim_sys->CPU_PT.RemoveAllInTransferPages();
  }
  printf("  CPU PFQ: %ld, SSD PFQ: %ld, Alloc PFQ: %ld, CPU PreEvcQ: %ld, SSD PreEvcQ: %ld, PreallocQ: %ld, CPU PrefetchQ: %ld, SSD PrefetchQ: %ld\n",
        sim_sys->pf_CPU_queue.size(),
        sim_sys->pf_SSD_queue.size(),
        sim_sys->pf_alloc_queue.size(),
        sim_sys->preevict_CPU_queue.size(),
        sim_sys->preevict_SSD_queue.size(),
        sim_sys->prealloc_queue.size(),
        sim_sys->prefetch_CPU_queue.size(),
        sim_sys->prefetch_SSD_queue.size());
}
// KernelBeginEvent END ========================


// Stall model BEGIN

//Input: 
// t_00: Fully PF execution time from profiling (ms)
// t_10: (InputPF) Output Fully PF time from profiling (ms)
// t_11: no PF execution time from profiling (ms)
// r_input: the ratio of PF data in the input tensors (and not in-transfer)
// r_output: the ratio of PF data in the output tensors (and not in-transfer)
// r_input_ssd: the ratio of SSD PFs in the PFs (not in-transfer) in the input tensors
// r_output_ssd: the ratio of SSD PFs in the PFs (not in-transfer) in the output tensors
// s_input: total input tensor size (byte)
// s_output: total output tensor size (byte)
// BW_pcie: PCIe bandwidth (B/ms)
// BW_ssd: SSD bandwidth (B/ms)
// l_sys: System (CPU far-fault handling) latency (us)
// l_ssd: SSD latency (us)

//Output:
// delteT_PF: delta t for page fault handling (ms)
// BW_ssd_rest: The rest of SSD bandwidth useable when handling PF (for prefetching) (B/ms)
// BW_pcie_rest: The rest of PCIe bandwidth usable when handling PF (for prefetching) （B/ms）

void performance_model(double t_00, double t_10, double t_11, double r_input, double r_output, 
                       double r_input_ssd, double r_output_ssd, long s_input, long s_output, 
                       double BW_pcie, double BW_ssd, int l_sys, int l_ssd, 
                       double& deltaT_PF, double& BW_ssd_rest, double& BW_pcie_rest) {
    
    Assert(r_input >= 0 && r_input <= 1);
    Assert(r_output >=0 && r_output <= 1);
    Assert(r_input_ssd >= 0 && r_input_ssd <= 1);
    Assert(r_output_ssd >= 0 && r_output_ssd <= 1);

    double BW_cpu_in;
    double BW_cpu_out;

    if (t_00 <= t_11)
    {
        t_00 = t_11;
    }
    if (t_10 < t_11)
    {
        t_10 = t_11;
    }
    if (t_10 > t_00)
    {
        t_10 = t_00;
    }

    if (t_00 == t_10)
    {
        BW_cpu_in = 1063004400;
    }
    else
    {
        BW_cpu_in = s_input / (t_00 - t_10);
    }
    if (t_10 == t_11)
    {
        BW_cpu_out = 1063004400;
    }
    else
    {
        BW_cpu_out = s_output / (t_10 - t_11);
    }

    double BW_ssd_in = BW_cpu_in * l_sys / (l_sys + l_ssd);
    double BW_ssd_out = BW_cpu_out * l_sys / (l_sys + l_ssd);

    if (BW_ssd_in > BW_ssd)
    {
        BW_ssd_in = BW_ssd;
    }
    if (BW_ssd_out > BW_ssd)
    {
        BW_ssd_out = BW_ssd;
    }
    if (r_input_ssd == 0)
    {
        BW_ssd_in = 0;
    }
    if (r_output_ssd == 0)
    {
        BW_ssd_out = 0;
    }
    
    

    if (BW_cpu_in > BW_pcie - max(BW_ssd_in, BW_ssd_out))
    {
        BW_cpu_in = BW_pcie - max(BW_ssd_in, BW_ssd_out);
    }

    if (BW_cpu_out > BW_pcie - max(BW_ssd_in, BW_ssd_out))
    {
        BW_cpu_out = BW_pcie - max(BW_ssd_in, BW_ssd_out);
    }

    double ssd_input_pf_time;
    double ssd_output_pf_time;
    
    if (r_input_ssd==0)
    {
        ssd_input_pf_time = 0;
    }
    else
    {
        ssd_input_pf_time = r_input*r_input_ssd*s_input/BW_ssd_in;
    }

    if(r_output_ssd==0)
    {
        ssd_output_pf_time = 0;
    }
    else
    {
        ssd_output_pf_time = r_output*r_output_ssd*s_output/BW_ssd_out;
    }
    double cpu_pf_time = r_input*(1-r_input_ssd)*s_input/BW_cpu_in + r_output*(1-r_output_ssd)*s_output/BW_cpu_out;
    

    deltaT_PF = max(ssd_input_pf_time + ssd_output_pf_time, cpu_pf_time);

    BW_ssd_rest = (r_input_ssd*r_input==0 && r_output_ssd*r_output==0) ? BW_ssd : BW_ssd - max(BW_ssd_in, BW_ssd_out);
    BW_pcie_rest = (cpu_pf_time==0) ? BW_pcie : BW_pcie - max(BW_cpu_in, BW_cpu_out);

}


// BatcherEvent BEGIN ========================
pair<int, int> BatcherEvent::processFetch(Addr start_addr, PageLocation src, bool is_pf) {
  // <pg_move_in, pg_move_out>
  // continue to alloc it in GPU
  pair<int, int> alloc_info = processAlloc(start_addr, is_pf);
  if (alloc_info.first < 0) {
    Assert(!is_pf);
    return alloc_info;
  }
  // record and afterprocess everything
  recordFetch(src, alloc_info.first);
  CPUPageTable::CPUPageTableEntry *CPU_PTE = sim_sys->CPU_PT.getEntry(start_addr);
  Assert(CPU_PTE);
  // PCIe transfer instant
  sim_sys->GPU_PT.markArrivedPTE(start_addr); 
  sim_sys->CPU_PT.markArrivedPTE(start_addr);
  sim_sys->CPU_PT.RemoveInTransferPage(start_addr);
  return alloc_info;
}

pair<int, int> BatcherEvent::processAlloc(Addr start_addr, bool is_pf) {
  // <allocate_pg_num, outgoing_pg_num>
  CPUPageTable::CPUPageTableEntry *CPU_PTE = sim_sys->CPU_PT.getEntry(start_addr);
  Assert(CPU_PTE);
  // mark allocation for the page in GPU PT and determine possible eviction
  pair<int, int> alloc_info = make_pair(1, 0);
  if (sim_sys->GPU_PT.allocPTE(start_addr)) {
    // free slot in GPU PT
    alloc_info.second = 0;
    // remove the page in CPU and reset force eviction destination if necessary
    if (CPU_PTE->location == PageLocation::IN_CPU) {
      sim_sys->CPU_PT.erasePTE(start_addr);
      if (forced_evc_dest == PageLocation::IN_SSD)
        forced_evc_dest = PageLocation::NOT_KNOWN;
    }
    sim_sys->GPU_PT.markArrivedPTE(start_addr);
    sim_sys->CPU_PT.markArrivedPTE(start_addr);
  } else {
    // redefine PF as real PF, even if it comes from prefetch/prealloc
    is_pf = is_pf || sim_sys->pageIsRequired(start_addr);
    // get eviction suggestion
    // <vpn, GPU_PTE, destination, candidate_info>
    auto evict_entry = sim_sys->GPU_PT.selectEvictPTE(
        sim_sys->getCurrentKernel()->kernel_id, is_pf);
    Addr vpn = get<0>(evict_entry);
    PageLocation destination = std::get<2>(evict_entry);
    Tensor *candidate_tensor = std::get<3>(evict_entry).tensor;
    // DEEPUM specific handling
    if (candidate_tensor == nullptr) {
      // cannot find a target tensor, stalling prefetch
      Assert(!is_pf);
      alloc_info.first = -1;
      alloc_info.second = -1;
      return alloc_info;
    }
    Assert(!sim_sys->CPU_PT.getEntry(vpn)->in_transfer);
    // destination override
    if (destination != PageLocation::NOT_PRESENT && forced_evc_dest != PageLocation::NOT_KNOWN) 
      destination = forced_evc_dest;
    alloc_info.second = processEvict(vpn, destination, true);
    // sanity check
    Assert(candidate_tensor);
    bool living = candidate_tensor->is_alive(sim_sys->getCurrentKernel()->kernel_id);
    Assert(!(living && destination == NOT_PRESENT));
    // remove the page in CPU and reset force eviction destination if necessary
    if (CPU_PTE->location == PageLocation::IN_CPU) {
      sim_sys->CPU_PT.erasePTE(start_addr);
      if (forced_evc_dest == PageLocation::IN_SSD)
        forced_evc_dest = PageLocation::NOT_KNOWN;
    }
    // alloc new page in GPU PT
    Assert(sim_sys->GPU_PT.allocPTE(start_addr));
    sim_sys->GPU_PT.markArrivedPTE(start_addr);
    sim_sys->CPU_PT.markArrivedPTE(start_addr);

    // sim_stat->addEvcSelection(sim_sys->getCurrentIteration(),
    //                           scheduled_time,
    //                           sim_sys->getCurrentKernel()->kernel_id,
    //                           destination,
    //                           std::get<3>(evict_entry));
  }
  GPUPageTable::GPUPageTableEntry *GPU_PTE = sim_sys->GPU_PT.getEntry(start_addr);
  Assert(GPU_PTE);
  // mark allocation for the page in CPU PT
  CPU_PTE->location = IN_GPU;
  CPU_PTE->ppn = GPU_PTE->ppn;
  sim_sys->CPU_PT.RemoveInTransferPage(start_addr);
  Assert(!CPU_PTE->in_transfer && !GPU_PTE->alloced_no_arrival);
  return alloc_info;
}

size_t BatcherEvent::processEvict(Addr start_addr, PageLocation dest, bool is_pf) {
  CPUPageTable::CPUPageTableEntry *CPU_PTE = sim_sys->CPU_PT.getEntry(start_addr);
  Assert(CPU_PTE);
  Assert(is_pf || CPU_PTE->in_transfer);

  size_t evc_count = 0;
  Tensor *tensor = sim_sys->GPU_PT.searchTensorForPage(start_addr);
  bool living = tensor->is_alive(sim_sys->getCurrentKernel()->kernel_id);
  if (living && dest == NOT_PRESENT) {
    if (is_pf) {
      wprintf("  Batcher [PF EVICTING LIVING PAGE] "
          "Kernel: %d Tensor: %d Addr: %ld Loc: %s Living: %d-%d\n",
          sim_sys->getCurrentKernel()->kernel_id, tensor->tensor_id, 
          start_addr, print_pagelocation_array[dest].c_str(),
          tensor->live_interval[0], tensor->live_interval[1]);
    } else {
      eprintf("  Batcher Kernel: %d Tensor: %d Addr: %ld Loc: %s Living: %d-%d\n", 
          sim_sys->getCurrentKernel()->kernel_id, tensor->tensor_id, 
          start_addr, print_pagelocation_array[dest].c_str(),
          tensor->live_interval[0], tensor->live_interval[1]);
      Assert(false);
    }
  }

  if (dest == IN_CPU) {
    sim_sys->CPU_PT.allocPTE(start_addr);
    Assert(CPU_PTE->location == IN_CPU);
    evc_count = 1;
  } else if (dest == IN_SSD) {
    CPU_PTE->location = IN_SSD;
    evc_count = 1;
  } else if (dest == NOT_PRESENT) {
    CPU_PTE->location = NOT_PRESENT;
    evc_count = 0;
  } else {
    Assert(false);
  }
  recordEvict(dest, evc_count);
  // re-enter PF queue if required by current kernel
  if (sim_sys->tensorIsRequired(tensor)) {
    // do not mark off in_transfer, it would be put back to PF lists
    if (dest == IN_CPU) {
      sim_sys->pf_CPU_queue.push_back(start_addr);
    } else if (dest == IN_SSD) {
      sim_sys->pf_SSD_queue.push_back(start_addr);
    } else if (dest == NOT_PRESENT) {
      sim_sys->pf_alloc_queue.push_back(start_addr);
    } else {
      Assert(false);
    }
    sim_sys->CPU_PT.markInTransferPTE(start_addr);
    sim_sys->CPU_PT.AddInTransferPages(start_addr);
  } else {
    // PCIe transfer instant, page arrives
    sim_sys->CPU_PT.markArrivedPTE(start_addr);
  }
  sim_sys->GPU_PT.erasePTE(start_addr);
  return evc_count;
}

void BatcherEvent::processPreevict() {
  while (outgoing_pg_SSD < sim_sys->SSD_PCIe_batch_num &&
         outgoing_pg_num < sim_sys->GPU_PCIe_batch_num &&
         sim_sys->preevict_SSD_queue.size()) {
      Addr start_addr = sim_sys->preevict_SSD_queue.front();
      Assert(forced_evc_dest == PageLocation::NOT_KNOWN ||
             (sim_sys->CPU_PT.reachMemoryLine() && forced_evc_dest == PageLocation::IN_SSD));
      PageLocation destination = forced_evc_dest == PageLocation::NOT_KNOWN ? PageLocation::IN_SSD : forced_evc_dest;
      size_t out = processEvict(start_addr, destination, false);
      sim_sys->preevict_SSD_queue.pop_front();
  }
  while (((sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_SSD < sim_sys->SSD_PCIe_batch_num) ||
          (!sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_CPU < sim_sys->CPU_PCIe_batch_num)) &&
         outgoing_pg_num < sim_sys->GPU_PCIe_batch_num &&
         sim_sys->preevict_CPU_queue.size()) {
      Addr start_addr = sim_sys->preevict_CPU_queue.front();
      Assert(forced_evc_dest == PageLocation::NOT_KNOWN ||
             (outgoing_pg_SSD == sim_sys->SSD_PCIe_batch_num && forced_evc_dest == PageLocation::IN_CPU) ||
             (sim_sys->CPU_PT.reachMemoryLine() && forced_evc_dest == PageLocation::IN_SSD));
      PageLocation destination = forced_evc_dest == PageLocation::NOT_KNOWN ? PageLocation::IN_CPU : forced_evc_dest;
      size_t out = processEvict(start_addr, destination, false);
      sim_sys->preevict_CPU_queue.pop_front();
  }
}

void BatcherEvent::processPFFetch(deque<Addr>* queue) {
  PageLocation src = NOT_KNOWN;
  if (queue == &sim_sys->pf_CPU_queue) {
    src = PageLocation::IN_CPU;
  } else if (queue == &sim_sys->pf_SSD_queue) {
    src = PageLocation::IN_SSD;
  } else if (queue == &sim_sys->prefetch_CPU_queue) {
    src = PageLocation::IN_CPU;
    Assert(sim_sys->mig_policy == MigPolicy::DEEPUM);
  } else if (queue == &sim_sys->prefetch_SSD_queue) {
    src = PageLocation::IN_SSD;
    Assert(sim_sys->mig_policy == MigPolicy::DEEPUM);
  } else {
    Assert(false);
  }
      
  while (incoming_pg_num < sim_sys->GPU_PCIe_batch_num &&
         alloc_pg_num < sim_sys->alloc_batch_num &&
         ((sim_sys->GPU_PT.isFull() && ((!sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num) ||
                                        (sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num && 
                                         outgoing_pg_SSD < sim_sys->SSD_PCIe_batch_num))) ||
          !sim_sys->GPU_PT.isFull()) &&
         queue->size()) {
    Addr start_addr = queue->front();
    queue->pop_front();

    CPUPageTable::CPUPageTableEntry *CPU_PTE = sim_sys->CPU_PT.getEntry(start_addr);
    Assert(CPU_PTE->location != IN_GPU);
    // <incoming_pg_num, outgoing_pg_num>
    pair<int, int> fetch_info = processFetch(start_addr, src, true);
    Assert(fetch_info.first >= 0);
    Assert(fetch_info.second >= 0);
    // fetch limit reached
    if (forced_fetch_src != NOT_KNOWN && forced_fetch_src != src)
      break;
  }
}

void BatcherEvent::processPrefetch() {
  // SSD prefetch handle
  while (incoming_pg_SSD < sim_sys->SSD_PCIe_batch_num &&
         incoming_pg_num < sim_sys->GPU_PCIe_batch_num &&
         ((sim_sys->GPU_PT.isFull() && ((!sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num) ||
                                        (sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num && 
                                         outgoing_pg_SSD < sim_sys->SSD_PCIe_batch_num))) ||
          !sim_sys->GPU_PT.isFull()) &&
         sim_sys->prefetch_SSD_queue.size()) {
    Addr start_addr = 0;
    // get target starting address
    if (sim_sys->mig_policy == MigPolicy::DEEPUM)    start_addr = sim_sys->prefetch_SSD_queue.front();
    else if (sim_sys->mig_policy == MigPolicy::OURS) start_addr = sim_sys->prefetch_SSD_queue.front();
    else Assert(false);
    // handle the target page
    CPUPageTable::CPUPageTableEntry *CPU_PTE = sim_sys->CPU_PT.getEntry(start_addr);
    Assert(CPU_PTE->location != IN_GPU);
    PageLocation real_from = CPU_PTE->location;
    if (real_from == IN_SSD) {
      pair<int, int> fetch_info = processFetch(start_addr, real_from, false);
      // stalling by direct return
      if (fetch_info.first < 0) {
        // iprintf("SSD prefetch queue stalled bc eviction candidate not found\n", "");
        return;
      }
    } else if (real_from == IN_CPU) {
      sim_sys->prefetch_CPU_queue.push_back(start_addr);
    }
    // pop target starting address, indicating processing done
    if (sim_sys->mig_policy == MigPolicy::DEEPUM)    sim_sys->prefetch_SSD_queue.pop_front();
    else if (sim_sys->mig_policy == MigPolicy::OURS) sim_sys->prefetch_SSD_queue.pop_front();
    else Assert(false);
  }
  // CPU prefetch handle
  while (incoming_pg_CPU < sim_sys->CPU_PCIe_batch_num &&
         incoming_pg_num < sim_sys->GPU_PCIe_batch_num &&
         ((sim_sys->GPU_PT.isFull() && ((!sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num) ||
                                        (sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num && 
                                         outgoing_pg_SSD < sim_sys->SSD_PCIe_batch_num))) ||
          !sim_sys->GPU_PT.isFull()) &&
         sim_sys->prefetch_CPU_queue.size()) {
    Addr start_addr = 0;
    // get target starting address
    if (sim_sys->mig_policy == MigPolicy::DEEPUM)    start_addr = sim_sys->prefetch_CPU_queue.front();
    else if (sim_sys->mig_policy == MigPolicy::OURS) start_addr = sim_sys->prefetch_CPU_queue.front();
    else Assert(false);
    // handle the target page
    CPUPageTable::CPUPageTableEntry *CPU_PTE = sim_sys->CPU_PT.getEntry(start_addr);
    Assert(CPU_PTE->location != IN_GPU);
    PageLocation real_from = CPU_PTE->location;
    if (real_from == IN_CPU) { 
      pair<int, int> fetch_info = processFetch(start_addr, real_from, false);
      // stalling by direct return
      if (fetch_info.first < 0) {
        // iprintf("CPU prefetch queue stalled bc eviction candidate not found\n", "");
        return;
      }
    } else if (real_from == IN_SSD) {
      sim_sys->prefetch_SSD_queue.push_back(start_addr);
    }
    // pop target starting address, indicating processing done
    if (sim_sys->mig_policy == MigPolicy::DEEPUM)    sim_sys->prefetch_CPU_queue.pop_front();
    else if (sim_sys->mig_policy == MigPolicy::OURS) sim_sys->prefetch_CPU_queue.pop_front();
    else Assert(false);
  }
}

void BatcherEvent::processPFAlloc(deque<Addr>* queue) {
  while (((sim_sys->GPU_PT.isFull() && ((!sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num) ||
                                        (sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num && 
                                         outgoing_pg_SSD < sim_sys->SSD_PCIe_batch_num))) ||
          !sim_sys->GPU_PT.isFull()) &&
         alloc_pg_num < sim_sys->alloc_batch_num &&
         queue->size()) {
    Addr start_addr = queue->front();
    
    CPUPageTable::CPUPageTableEntry *CPU_PTE = sim_sys->CPU_PT.getEntry(start_addr);
    Assert(CPU_PTE->location != IN_GPU);
    // <allocate_pg_num, outgoing_pg_num>
    pair<int, int> alloc_info = processAlloc(start_addr, true);
    if (alloc_info.first < 0) {
      // stalled, do nothing
      // wprintf("Alloc event stalled bc GPU is filled with useful data\n", "");
      break;
    }
    alloc_pg_num += alloc_info.first;
    queue->pop_front();
  }
}

void BatcherEvent::processAlloc(bool is_pf) {
  deque<Addr>& queue = is_pf ? sim_sys->pf_alloc_queue : sim_sys->prealloc_queue;
  while (((sim_sys->GPU_PT.isFull() && ((!sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num) ||
                                        (sim_sys->CPU_PT.reachMemoryLine() && outgoing_pg_num < sim_sys->GPU_PCIe_batch_num && 
                                         outgoing_pg_SSD < sim_sys->SSD_PCIe_batch_num))) ||
          !sim_sys->GPU_PT.isFull()) &&
         alloc_pg_num < sim_sys->alloc_batch_num &&
         queue.size()) {
    Addr start_addr = queue.front();
    
    CPUPageTable::CPUPageTableEntry *CPU_PTE = sim_sys->CPU_PT.getEntry(start_addr);
    Assert(CPU_PTE->location != IN_GPU);
    // <allocate_pg_num, outgoing_pg_num>
    pair<int, int> alloc_info = processAlloc(start_addr, is_pf);
    if (alloc_info.first < 0) {
      // stalled, do nothing
      // wprintf("Alloc event stalled bc GPU is filled with useful data\n", "");
      break;
    }
    alloc_pg_num += alloc_info.first;
    queue.pop_front();
  }
}

void BatcherEvent::recordFetch(PageLocation src, size_t pg_num) {
  if (pg_num == 0) return;
  Assert(forced_fetch_src == PageLocation::NOT_KNOWN || forced_fetch_src == src);
  // TODO: logic need to be refined if multiple pages are used in the future
  if (src == IN_SSD) {
    incoming_pg_SSD += pg_num;
    Assert(incoming_pg_SSD <= sim_sys->SSD_PCIe_batch_num);
    if (incoming_pg_SSD == sim_sys->SSD_PCIe_batch_num) forced_fetch_src = IN_CPU;
  } else if (src == IN_CPU) {
    incoming_pg_CPU += pg_num;
    Assert(incoming_pg_CPU <= sim_sys->CPU_PCIe_batch_num);
    if (incoming_pg_CPU == sim_sys->CPU_PCIe_batch_num) forced_fetch_src = IN_SSD;
  } else {
    Assert(false);
  }
  alloc_pg_num += pg_num;
  incoming_pg_num += pg_num;
}

void BatcherEvent::recordEvict(PageLocation dest, size_t pg_num) {
  if (pg_num == 0) return;
  Assert(forced_evc_dest == PageLocation::NOT_KNOWN || forced_evc_dest == dest);
  // TODO: logic need to be refined if multiple pages are used in the future
  if (dest == IN_SSD) {
    outgoing_pg_SSD += pg_num;
    outgoing_pg_num += pg_num;
    Assert(outgoing_pg_SSD <= sim_sys->SSD_PCIe_batch_num);
    if (outgoing_pg_SSD == sim_sys->SSD_PCIe_batch_num) 
      forced_evc_dest = IN_CPU;
  } else if (dest == IN_CPU) {
    outgoing_pg_CPU += pg_num;
    outgoing_pg_num += pg_num;
    Assert(outgoing_pg_CPU <= sim_sys->CPU_PCIe_batch_num);
    if (outgoing_pg_CPU == sim_sys->CPU_PCIe_batch_num || sim_sys->CPU_PT.reachMemoryLine()) 
      forced_evc_dest = IN_SSD;
  } else {
    Assert(false);
  }
  Assert(outgoing_pg_num <= sim_sys->GPU_PCIe_batch_num);
}

bool BatcherEvent::shouldExecute() {
  return true;
}

void BatcherEvent::execute(vector<Event *> &created_events) {
  if (sim_sys->prefetch_CPU_queue.size() != 0 || 
      sim_sys->prefetch_SSD_queue.size() != 0 || 
      sim_sys->pf_CPU_queue.size() != 0 || 
      sim_sys->pf_SSD_queue.size() != 0 || 
      sim_sys->pf_alloc_queue.size() != 0 ||
      sim_sys->prealloc_queue.size() != 0 || 
      sim_sys->preevict_CPU_queue.size() != 0 || 
      sim_sys->preevict_SSD_queue.size() != 0) {
    if (sim_sys->CPU_PT.reachMemoryLine())
      forced_evc_dest = IN_SSD;
    
    // Process pre-eviction first
    processPreevict();
    // Further passive eviction is done with prefetch and prealloc
    
    // SSD PF fetch first
    processPFFetch(&sim_sys->pf_SSD_queue);
    // CPU PF fetch after SSD
    processPFFetch(&sim_sys->pf_CPU_queue);

    // PF alloc first
    processAlloc(true);
    // Guided transfer - Prefetch
    processPrefetch();
    // Prealloc after PF alloc
    processAlloc(false);

    // DeepUM specific lockup detection
    if (sim_sys->mig_policy == MigPolicy::DEEPUM &&
        sim_sys->reschedule_info && !sim_sys->reschedule_info->kernel_started &&
        sim_sys->CPU_PT.haveInTransferPages() && incoming_pg_num + outgoing_pg_num == 0) {
      processPFFetch(&sim_sys->prefetch_SSD_queue);
      processPFFetch(&sim_sys->prefetch_CPU_queue);
      processPFAlloc(&sim_sys->prealloc_queue);
    }

    if (sim_sys->batcher_evt_print_current == 0) {
      printf("Executing Bactcher Event @ %ld [Visible Interval: %d]\n", scheduled_time, sim_sys->batcher_evt_print_max);
      printf("  GPT: %10lu/%10lu   CPT %10lu/%10lu   In transfer %10ld\n"
              "    prefetchCPUq: %8lu prefetchSSDq: %8lu pfCPUq: %8lu pfSSDq: %8lu pfALCq: %8lu preALCq: %8lu preevcCPUq: %8lu preevcSSDq: %8lu"
              "    [IN %2lu = CPU %2lu + SSD %2lu, Alloc %4lu] [OUT %2lu = CPU %2lu + SSD %2lu]\n",
          sim_sys->GPU_PT.getCapacity().first, sim_sys->GPU_PT.getCapacity().second, 
          sim_sys->CPU_PT.getCapacity().first, sim_sys->CPU_PT.getCapacity().second,
          sim_sys->CPU_PT.numInTransferPages(), 
          sim_sys->prefetch_CPU_queue.size(), sim_sys->prefetch_SSD_queue.size(), 
          sim_sys->pf_CPU_queue.size(), sim_sys->pf_SSD_queue.size(), sim_sys->pf_alloc_queue.size(), 
          sim_sys->prealloc_queue.size(), 
          sim_sys->preevict_CPU_queue.size(), sim_sys->preevict_SSD_queue.size(), 
          incoming_pg_num, incoming_pg_CPU, incoming_pg_SSD, alloc_pg_num,
          outgoing_pg_num, outgoing_pg_CPU, outgoing_pg_SSD);
    }

    if (sim_sys->batcher_evt_print_current++ > sim_sys->batcher_evt_print_max) 
      sim_sys->batcher_evt_print_current = 0;

    if (!sim_sys->data_transferring) {
      sim_stat->addTransferBoundary(sim_sys->getCurrentIteration(),
                                    scheduled_time,
                                    true);
      sim_sys->data_transferring = true;
    }
  } else {
    if (sim_sys->data_transferring) {
      sim_stat->addTransferBoundary(sim_sys->getCurrentIteration(),
                                    scheduled_time,
                                    false);
      sim_sys->data_transferring = false;
    }
  }

  Assert(incoming_pg_SSD <= sim_sys->SSD_PCIe_batch_num);
  Assert(incoming_pg_CPU <= sim_sys->CPU_PCIe_batch_num);
  Assert(incoming_pg_num <= sim_sys->GPU_PCIe_batch_num);
  Assert(outgoing_pg_SSD <= sim_sys->SSD_PCIe_batch_num);
  Assert(outgoing_pg_CPU <= sim_sys->CPU_PCIe_batch_num);
  Assert(outgoing_pg_num <= sim_sys->GPU_PCIe_batch_num);
  sim_stat->addPCIeBWStat(sim_sys->getCurrentIteration(), scheduled_time,
                          incoming_pg_num, incoming_pg_SSD, incoming_pg_CPU,
                          outgoing_pg_num, outgoing_pg_SSD, outgoing_pg_CPU,
                          alloc_pg_num);

  // reschedule of kernel required
  if (sim_sys->reschedule_info && 
      sim_sys->pf_alloc_queue.size() == 0 &&
      sim_sys->pf_CPU_queue.size() == 0 &&
      sim_sys->pf_SSD_queue.size() == 0 &&
      !sim_sys->CPU_PT.haveInTransferPages()) {
    created_events.push_back(
        new KernelBeginEvent(scheduled_time, sim_sys->getCurrentKernel(),
                             sim_sys->reschedule_info->first_scheduled_time,
                             sim_sys->reschedule_info->page_faulted_time));
    sim_sys->reschedule_info->kernel_started = true;
    sim_sys->CPU_PT.RemoveAllInTransferPages();
  }
  // always create a new batcher event
  created_events.push_back(
      new BatcherEvent(scheduled_time + sim_sys->PCIe_batch_ii_cycle));
}
// BatcherEvent END ========================


} // namespace Simulator
