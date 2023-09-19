/* File: ast.h
 * ----------- 
 * This file defines the abstract base class Node and the concrete 
 * Identifier and Error node subclasses that are used through the tree as 
 * leaf nodes. A parse tree is a hierarchical collection of ast nodes (or, 
 * more correctly, of instances of concrete subclassses such as VarDecl,
 * ForStmt, and AssignExpr).
 * 
 * Location: Each node maintains its lexical location (line and columns in 
 * file), that location can be NULL for those nodes that don't care/use 
 * locations. The location is typcially set by the node constructor.  The 
 * location is used to provide the context when reporting semantic errors.
 *
 * Parent: Each node has a pointer to its parent. For a Program node, the 
 * parent is NULL, for all other nodes it is the pointer to the node one level
 * up in the parse tree.  The parent is not set in the constructor (during a 
 * bottom-up parse we don't know the parent at the time of construction) but 
 * instead we wait until assigning the children into the parent node and then 
 * set up links in both directions. The parent link is typically not used 
 * during parsing, but is more important in later phases.
 *
 * Printing: The only interesting behavior of the node classes for pp2 is the 
 * bility to print the tree using an in-order walk.  Each node class is 
 * responsible for printing itself/children by overriding the virtual 
 * PrintChildren() and GetPrintNameForNode() methods. All the classes we 
 * provide already implement these methods, so your job is to construct the
 * nodes and wire them up during parsing. Once that's done, printing is a snap!

 */

#ifndef _H_ast
#define _H_ast

#include <stdlib.h>   // for NULL
#include <string>
#include <vector>
#include "location.h"
#include "utility.h"
#include "list.h"



typedef enum {
  Conv2d_T, ReLU_T, MaxPool2d_T, AdaptiveAvgPool2d_T, Linear_T, Dropout_T, BatchNorm2d_T, Init_T, Add_T, Concat_T, Scale_T
} OperatorType;



class Node 
{
  protected:
    yyltype *location;
    Node *parent;

  public:
    Node(yyltype loc);
    Node();
    
    yyltype *GetLocation()   { return location; }
    void SetParent(Node *p)  { parent = p; }
    Node *GetParent()        { return parent; }

    virtual const char *GetPrintNameForNode() = 0;
    
    // Print() is deliberately _not_ virtual
    // subclasses should override PrintChildren() instead
    void Print(int indentLevel, const char *label = NULL); 
    virtual void PrintChildren(int indentLevel)  {}
};
   

class Identifier : public Node 
{
  protected:
    char *name;
    
  public:
    Identifier(yyltype loc, const char *name);
    const char *GetPrintNameForNode()   { return "Identifier"; }
    void PrintChildren(int indentLevel);
    char* GetIdName() { return name; }
};


// This node class is designed to represent a portion of the tree that 
// encountered syntax errors during parsing. The partial completed tree
// is discarded along with the states being popped, and an instance of
// the Error class can stand in as the placeholder in the parse tree
// when your parser can continue after an error.
class Error : public Node
{
  public:
    Error() : Node() {}
    const char *GetPrintNameForNode()   { return "Error"; }
};



//Nonterminal Operator
class Operatorr : public Node
{
  public:
    OperatorType type;
    Operatorr() : Node(), type(Init_T) {};
    Operatorr(yyltype loc) : Node(loc), type(Init_T) {};
};


class Block : public Node{
    public:
    virtual void Analysis() = 0;
};

class Layer : public Node
{
  public:
    Identifier* id;
    Layer(Identifier* name);
    virtual void Analysis() = 0;
};


class OperatorBlock : public Block
{
  public:
    Operatorr* op;

    OperatorBlock(Operatorr* op);
    const char *GetPrintNameForNode() { return "OperatorBlock"; }
    void PrintChildren(int indentLevel);
    virtual void Analysis();
};

class SeqBlock : public Node
{
  public:
    int number;
    Block* block;
    SeqBlock(int number, Block* block);
    const char *GetPrintNameForNode() { return "SeqBlock"; }
    void PrintChildren(int indentLevel);
    virtual void Analysis();
};


class UserBlock : public Block
{
  public:
    Identifier* name;
    List<Layer*> *layers;

    UserBlock(Identifier* id, List<Layer*> *layers);
    const char *GetPrintNameForNode() { return "UserBlock"; }
    void PrintChildren(int indentLevel);
   virtual  void Analysis();
};


class BlockLayer : public Layer
{
  public:
    Block* block;

    BlockLayer(Identifier* id, Block* b);
    const char *GetPrintNameForNode() { return "BlockLayer"; }
    void PrintChildren(int indentLevel);
    virtual void Analysis();
};


class SeqBlockLayer : public Layer
{
  public:
    List<SeqBlock*> *seqblocks;

    SeqBlockLayer(Identifier *id, List<SeqBlock*> *seqblocks);
    const char *GetPrintNameForNode() { return "SeqBlockLayer"; }
    void PrintChildren(int indentLevel);
    virtual void Analysis();
};

class Model : public Node
{
  public:
    Identifier* modelname;
    List<Layer*> *layers;

    Model(Identifier* name, List<Layer*> *layers);
    const char *GetPrintNameForNode() { return "Model"; }
    void PrintChildren(int indentLevel);
    virtual void Analysis();

};

class AddOp : public Operatorr
{
  public:
    AddOp(): Operatorr() { type = OperatorType::Add_T; };
    const char *GetPrintNameForNode()   { return "Add"; }
};


class ConcatOp : public Operatorr
{
  public:
    ConcatOp(): Operatorr() { type = OperatorType::Concat_T; };
    const char *GetPrintNameForNode()   { return "Concat"; }
};


class ScaleOp : public Operatorr
{
  public:
    ScaleOp(): Operatorr() { type = OperatorType::Scale_T; };
    const char *GetPrintNameForNode()   { return "Scale"; }
};


class Conv2d : public Operatorr
{
  public:
    int in_channels;
    int out_channels;
    int kernel_size_r;
    int kernel_size_s;
    int stride_0;
    int stride_1;
    int padding_0;
    int padding_1;
    bool bias;
    Conv2d(yyltype loc, int in_channels, int out_channels, int kernel_size_r, int kernel_size_s,
     int stride_x=1, int stride_y=1, int padding_x=0, int padding_y=0, bool bias=true);
    
    const char *GetPrintNameForNode()   { return "Conv2d"; }
    void PrintChildren(int indentLevel);

};

class ReLU : public Operatorr
{
  public:
    bool inplace;
    ReLU(yyltype loc, bool inplace=false);
    
    const char *GetPrintNameForNode()   { return "ReLU"; }
    void PrintChildren(int indentLevel);
};

class MaxPool2d : public Operatorr
{
  public:
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    bool ceil_mode;
    MaxPool2d(yyltype loc, int k, int s, int p, int d, bool ceil_mode);
    
    const char *GetPrintNameForNode()   { return "MaxPool2d"; }
    void PrintChildren(int indentLevel);
};


class AdaptiveAvgPool2d : public Operatorr
{
  public:
    int outputsize_0;
    int outputsize_1;
    AdaptiveAvgPool2d(yyltype loc, int ox, int oy);

    const char *GetPrintNameForNode()   { return "AdaptiveAvgPool2d"; }
    void PrintChildren(int indentLevel);
};


class Linear : public Operatorr
{
  public:
    int in_features;
    int out_features;
    bool bias;
    Linear(yyltype loc, int i, int o, bool b);

    const char *GetPrintNameForNode()   { return "Linear"; }
    void PrintChildren(int indentLevel);
};


class Dropout : public Operatorr
{
  public:
    double p;
    bool inplace;
    Dropout(yyltype loc, double p, bool i);

    const char *GetPrintNameForNode()   { return "Dropout"; }
    void PrintChildren(int indentLevel);
};


class BatchNorm2d : public Operatorr
{
  public:
    int num_features;
    double eps;
    double momentum;
    bool affline;
    bool track_running_stats;
    BatchNorm2d(yyltype loc, int n, double e, double m, bool a, bool tr);

    const char *GetPrintNameForNode()   { return "BatchNorm2d"; }
    void PrintChildren(int indentLevel);
};


class Hidding_Interval;

class Tensor
{
    private:
        Tensor();
    public:
        Tensor(long long size, bool glob = false);
        unsigned long getGlobalOffset();
        std::string name() const;
        bool is_alive(int current_kernel) const;
        void print() const;
        void print_liveness();
        void print_intervals();

        int tensor_id;
        long long size_in_byte;
        long long raw_size_byte; 
        long long address_offset;
        bool is_global_weight;
        bool is_choosed_to_evict = false;
        int live_interval[2]; //live_interval[0] = birth; live_interval[1] = death; if death=-1, it means that this tensor is always dead
        std::vector<Hidding_Interval*> hidding_intervals;

    //Flashneuron only: (starts with 'f')
        bool f_is_allocated_on_GPU = false;
        bool f_is_choosed_to_offload = false;
        bool f_is_fetching = false;
        long f_page_range[2];
};

class Hidding_Interval
{
  public:
    double time_estimated;   //us
    int kernelLevel_interval[2];
    int original_prefetch_index;
    int evict_finish_index;
    bool is_looped;
    bool is_offloaded;
    bool is_really_offloaded;
    long GPU_mem_line;
    Tensor* the_tensor;
    Hidding_Interval(Tensor* t, long GPU_line){the_tensor = t; is_looped = false; is_offloaded = false; is_really_offloaded =false; GPU_mem_line = GPU_line; original_prefetch_index = -1; evict_finish_index = -1;};
    void print();
};




class Model_Layer
{
  public:
    int layer_id;   // layer id is 1-based.
    int N;
    int C;
    int H;
    int W;

    //Concat
    std::vector<int> input_Cs;

    //Scale
    int scale_H;
    int scale_W;
    Model_Layer* get_scale_num_layer;

    //Data flow dependency
    std::vector<Model_Layer*> next_layers;
    std::vector<Model_Layer*> previous_layers;

    Tensor* input_activation = nullptr;
    Tensor* output_activation = nullptr;
    Tensor* weight = nullptr;
    Tensor* d_input = nullptr;
    Tensor* d_output = nullptr;
    Tensor* d_weight = nullptr;
    Tensor* bias = nullptr;
    Tensor* d_bias = nullptr;

    //BatchNorm:
    Tensor* alpha_and_beta = nullptr;
    Tensor* d_alpha_and_beta = nullptr;
    Tensor* running_m = nullptr;
    Tensor* running_v = nullptr;

    //BatchNorm working space
    Tensor* mu = nullptr;
    Tensor* var = nullptr;
    Tensor* v1 = nullptr;
    Tensor* v2 = nullptr;
    Tensor* d_mu = nullptr;
    Tensor* d_var = nullptr;
    Tensor* d_v1 = nullptr;
    Tensor* d_v2 = nullptr;

    //Dropout:
    Tensor* musk_array = nullptr;

    //Add and concat:
    std::vector<Tensor*> other_inputs; //Aligned to "previous_layers[1-:]"
    std::vector<Tensor*> other_d_inputs; //Aligned to "previous_layers[1-:]"

    //For any branch layers:
    std::vector<Tensor*> other_d_outputs; //Aligned to "next_layers[1-:]"

    bool inplace;

    Operatorr* operatorr;
    Model_Layer();
    Model_Layer(Operatorr* op, int id);
    void give_next_layer_size (int* N, int* C, int* H, int* W);
    void print_name();
    void print();
    std::string get_print_msg();
};


// Transformers from Microsoft


struct OP_tensor
{
  public:
    int dim;
    int tensor_id;
    bool is_const = false;
    Tensor* tensor = nullptr;
    Tensor* d_tensor = nullptr;
    std::vector<int> dims;
};

typedef enum{
  GatherV2, Dot, Relu, Add, BatchMatMul, Divide, Multiply, Power, SoftmaxBasic,
   Sqrt, Subtract, Sum, Tanh, Convolution, MaxPool5, Erf
} ModelOP_type;

class Model_OP
{
  public:
    int op_id;   // op id is 0-based.
    std::string type;
    int N;  //batch_size
    //int C;  //hidden_width
    
    int output_dim;
    std::vector<int> output_dims;

    int input_num;

    std::vector<OP_tensor> input_tensors;

    Tensor* output_tensor = nullptr;
    std::vector<Tensor*> d_output_tensors;


    // //Data flow dependency
    // std::vector<Model_OP*> next_layers;
    // std::vector<Model_OP*> previous_layers;

  
    // Model_OP();
    
    // void print_name();
    void print();

};

void transformer_parse(std::string filename);


#endif
