#ifndef __SIMULATION_EVENTS_H__
#define __SIMULATION_EVENTS_H__

#include "simulationComponents.h"
#include "analysis.h"

using std::vector;
using std::string;
using std::to_string;

namespace Simulator {

/**
 * @brief abstraction of event class, all the custom events must be inherited from this
 *        class
 */
class Event {
  public:
    /**
     * @brief construct an event using its scheduled time and its name
     * @param scheduled_time the desired time of executing this event
     * @param name name of event, ease to print in human-readable format
     */
    Event(unsigned long scheduled_time, string name) :
        scheduled_time(scheduled_time), name(name) {}

    virtual bool shouldExecute() = 0;
    /**
     * @brief execution of the event
     * @param created_events list of events that are created during the execution of
     *                       this event
     * @note on the execution of the event, created_events is asserted to be empty
     *       function will be executed when the event is pop out of the event queue
     *       all the child class need to implement this pure virtual function
     */
    virtual void execute(vector<Event *> &created_events) = 0;

    virtual ~Event() {};

    // Scheduled time of this event. When the execute function is executed
    // this is also the current time.
    const unsigned long scheduled_time;
    // name for identifying the event
    const string name;
};

/**
 * @brief event pointer wrapper used to compare events based upon their scheduled time
 */
struct EventPtr {
  EventPtr(Event *ptr) : ptr(ptr) {}
  Event *ptr;
};

/**
 * @brief comparator class used to compare events based upon their scheduled time
 */
struct EventPtrComp {
  bool operator()(const EventPtr &lhs, const EventPtr &rhs) const {
    Assert(lhs.ptr && rhs.ptr);
    return lhs.ptr->scheduled_time > rhs.ptr->scheduled_time;
  }
};

/**
 * @brief event that marks the beginning of execution of an event
 */
class KernelBeginEvent : public Event {
  public:
    /**
     * @brief construct a new kernel event if is not rescheduled
     * @param scheduled_time starting execution time of the kernel
     * @param kernel cuda kernel that is executed
     */
    KernelBeginEvent(unsigned long scheduled_time, const CUDAKernel *kernel) :
        Event(scheduled_time, "Kernel" + to_string(kernel->kernel_id)),
        kernel(kernel) {}

    /**
     * @brief construct a new kernel event if is rescheduled
     * @param scheduled_time starting execution time of the kernel
     * @param kernel cuda kernel that is executed
     * @param first_scheduled_time
     * @param page_faulted_time
     */
    KernelBeginEvent(unsigned long scheduled_time, const CUDAKernel *kernel,
                     unsigned long first_scheduled_time,
                     unsigned long page_faulted_time) :
        Event(scheduled_time, "Kernel" + to_string(kernel->kernel_id)),
        kernel(kernel) {}
    ~KernelBeginEvent() {}

    bool shouldExecute();
    /**
     * @brief - When the tensors that are required by the kernel are not yet arrived in the
     *        GPU memory, it would create page faults. In this situation, the kernel
     *        simulated execution will be postponed, and an KernelRescheduleInfo object,
     *        System.reschedule_info, is populated for later invocations by the batcher
     *        - When the tensors that are required by the kernel are all present in the GPU
     *        memory, the kernel will be executed, and a new kernel will be scheduled after
     *        a simulated time delay.
     */
    void execute(vector<Event *> &created_events);

    // the cuda kernel that is processed in this event
    const CUDAKernel *kernel;
  private:
  /**
   * @brief check if all the required tensors have arrived at GPU
   * @param required_tensors all the tensors that are required by this kernel
   * @return whether all the pages required are in the GPU memory
   */
    bool requiredPageArrived(vector<Tensor *> &required_tensors, bool overtime);

    /**
     * @brief try to transfer all the pages for the tensor, all the pages that are requested
     *        are marked as critical
     * @param tensor the tensor to be processed
     * @return PageFaultInfo
     */
    PageFaultInfo transferTensorToGPU(Tensor *tensor, bool is_input);

    void guidedTransfer(DataMovementHint *hint);

    /**
     * @brief get the expected page faulted execution time of the kernel
     * @param info information about this page fault
     * @return time taken for this kernel to finish execution
     * @todo black box needed to be added
     */
    unsigned long getPageFaultTime(PageFaultInfo &info);
};

/**
 * @brief
 * @note 
 */
class BatcherEvent : public Event {
  public:
    BatcherEvent(unsigned long scheduled_time) :
        Event(scheduled_time, "Exec"),
        alloc_pg_num(0), incoming_pg_num(0), outgoing_pg_num(0),
        incoming_pg_SSD(0), incoming_pg_CPU(0),
        outgoing_pg_SSD(0), outgoing_pg_CPU(0),
        forced_fetch_src(PageLocation::NOT_KNOWN),
        forced_evc_dest(PageLocation::NOT_KNOWN) {}
    ~BatcherEvent() {}
    bool shouldExecute();
    void execute(vector<Event *> &created_events);
  private:
    void processPreevict();
    void processPFFetch(deque<Addr>* queue);
    void processPrefetch();
    void processPFAlloc(deque<Addr>* queue);
    void processAlloc(bool is_pf);

    pair<int, int> processFetch(Addr start_addr, PageLocation src, bool is_pf);
    pair<int, int> processAlloc(Addr start_addr, bool is_pf);
    size_t processEvict(Addr starting_addr, PageLocation dest, bool is_pf);

    void recordFetch(PageLocation dest, size_t pg_num);
    void recordEvict(PageLocation dest, size_t pg_num);

    size_t alloc_pg_num, incoming_pg_num, outgoing_pg_num;
    size_t incoming_pg_SSD, incoming_pg_CPU;
    size_t outgoing_pg_SSD, outgoing_pg_CPU;
    PageLocation forced_fetch_src, forced_evc_dest;
};


} // namespace Simulator

#endif