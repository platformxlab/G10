/* File: ast.cc
 * ------------
 */


#include "ast.h"
#include <string.h> // strdup
#include <stdio.h>  // printf
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>


int is_resnet = 0;
int is_inception = 0;
int is_senet = 0;
int batch_size;
int input_H;
int input_W;
extern int borden;
extern int is_transformer;
extern double loosen_parameter;
extern double SSD_PCIe_bandwidth_GBps;

std::vector<Model_Layer*> forward_layers;
int layer_id = 0;

std::vector<Model_OP*> forward_ops;
std::unordered_map<int, Model_OP*> op_map;
std::unordered_map<int, OP_tensor> transformer_tensors;

Node::Node(yyltype loc) {
    location = new yyltype(loc);
    parent = NULL;
}

Node::Node() {
    location = NULL;
    parent = NULL;
}

/* The Print method is used to print the parse tree nodes.
 * If this node has a location (most nodes do, but some do not), it
 * will first print the line number to help you match the parse tree 
 * back to the source text. It then indents the proper number of levels 
 * and prints the "print name" of the node. It then will invoke the
 * virtual function PrintChildren which is expected to print the
 * internals of the node (itself & children) as appropriate.
 */
void Node::Print(int indentLevel, const char *label) { 
    const int numSpaces = 3;
    printf("\n");
    if (GetLocation()) 
        printf("%*d", numSpaces, GetLocation()->first_line);
    else 
        printf("%*s", numSpaces, "");
    printf("%*s%s%s: ", indentLevel*numSpaces, "", 
           label? label : "", GetPrintNameForNode());
   PrintChildren(indentLevel);
} 
	 
Identifier::Identifier(yyltype loc, const char *n) : Node(loc) {
    name = strdup(n);
} 

void Identifier::PrintChildren(int indentLevel) {
    printf("%s", name);
}

Conv2d::Conv2d(yyltype loc, int in_channels, int out_channels, int kernel_size_r, int kernel_size_s,
     int stride_x, int stride_y, int padding_x, int padding_y, bool bias) : Operatorr(loc){
    type = OperatorType::Conv2d_T;
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size_r = kernel_size_r;
    this->kernel_size_s = kernel_size_s;
    this->stride_0 = stride_x;
    this->stride_1 = stride_y;
    this->padding_0 = padding_x;
    this->padding_1 = padding_y;
    this->bias = bias;
}

void Conv2d::PrintChildren(int indentLevel){
    printf("(%d, %d, %d, %d, %d, %d, %d, %d, %d)", in_channels, out_channels, kernel_size_r, kernel_size_s, stride_0, stride_1, padding_0, padding_1, bias);
}


ReLU::ReLU(yyltype loc, bool inplace) : Operatorr(loc){
    type = OperatorType::ReLU_T;
    this->inplace = inplace;
}

void ReLU::PrintChildren(int indentLevel){
    printf("(%d)", inplace);
}


MaxPool2d::MaxPool2d(yyltype loc, int k, int s, int p, int d, bool ceil_mode) : Operatorr(loc){
    type = OperatorType::MaxPool2d_T;
    this->kernel_size = k;
    this->stride =s;
    this->padding = p;
    this->dilation = d;
    this->ceil_mode = ceil_mode;
}

void MaxPool2d::PrintChildren(int indentLevel){
    printf("(%d, %d, %d, %d, %d)", kernel_size, stride, padding, dilation, ceil_mode);
}


AdaptiveAvgPool2d::AdaptiveAvgPool2d(yyltype loc, int ox, int oy) : Operatorr(loc){
    type = OperatorType::AdaptiveAvgPool2d_T;
    this->outputsize_0 = ox;
    this->outputsize_1 = oy;
}

void AdaptiveAvgPool2d::PrintChildren(int indentLevel){
    printf("(%d, %d)", outputsize_0, outputsize_1);
}


Linear::Linear(yyltype loc, int i, int o, bool b) : Operatorr(loc){
    type = OperatorType::Linear_T;
    this->in_features = i;
    this->out_features = o;
    this->bias = b;
}

void Linear::PrintChildren(int indentLevel){
    printf("(%d, %d, %d)", in_features, out_features, bias);
}

Dropout::Dropout(yyltype loc, double p, bool i) : Operatorr(loc){
    type = OperatorType::Dropout_T;
    this->p = p;
    this->inplace = i;
}


void Dropout::PrintChildren(int indentLevel){
    printf("(%f, %d)", p, inplace);
}


BatchNorm2d::BatchNorm2d(yyltype loc, int n, double e, double m, bool a, bool tr) : Operatorr(loc){
    type = OperatorType::BatchNorm2d_T;
    this->num_features = n;
    this->eps = e;
    this->momentum = m;
    this->affline = a;
    this->track_running_stats = tr;
}

void BatchNorm2d::PrintChildren(int indentLevel){
    printf("(%d, %f, %f, %d, %d)", num_features, eps, momentum, affline, track_running_stats);
}


OperatorBlock::OperatorBlock(Operatorr* op){
    Assert(op != NULL);
    this->op = op;
    this->op->SetParent(this);
}

void OperatorBlock::PrintChildren(int indentLevel){
    op->Print(indentLevel+1);
}

void OperatorBlock::Analysis(){
    Model_Layer* newop = new Model_Layer(op, layer_id);
    if (layer_id==0)
    {
        //TODO: use file-input here
        newop->N = batch_size;
        newop->H = input_H;
        newop->W = input_W;
        Assert(op->type==OperatorType::Conv2d_T);
        Conv2d* firstlayer = dynamic_cast<Conv2d*>(op);
        newop->C = firstlayer->in_channels;
    }
    else
    {
        //Default
        newop->previous_layers.push_back(forward_layers.back());
        forward_layers.back()->next_layers.push_back(newop);
    }
    forward_layers.push_back(newop);
    layer_id++;
}

SeqBlock::SeqBlock(int num, Block* block){
    Assert(block!=NULL);
    number = num;
    this->block = block;
    this->block->SetParent(this);
}

void SeqBlock::PrintChildren(int indentLevel){
    block->Print(indentLevel+1);
}

void SeqBlock::Analysis(){
    block->Analysis();
}


Layer::Layer(Identifier* id) : Node(*id->GetLocation()) {
    Assert(id != NULL);
    (this->id = id)->SetParent(this);
}


UserBlock::UserBlock(Identifier* id, List<Layer*> *layers){
    Assert(id != NULL && layers != NULL);
    this->name = id;
    this->name->SetParent(this);
    this->layers = layers;
    this->layers->SetParentAll(this);
}

void UserBlock::PrintChildren(int indentLevel){
    name->Print(indentLevel+1, "Blockname");
    layers->PrintAll(indentLevel+1, "Layers");
}

void UserBlock::Analysis(){
    for (int i = 0; i < layers->NumElements(); i++)
    {
        layers->Nth(i)->Analysis();
    }
    //TODO: Self-defined user block
    // Dependency
    if (is_resnet==1)
    {
        if (!strcmp(name->GetIdName(), "BasicBlock"))
        {
            if (layers->NumElements()==6) //It has downsample
            {
                //Create new "Add" layer
                Operatorr* add_op = new AddOp();
                Model_Layer* add_layer = new Model_Layer(add_op, layer_id);
                forward_layers.push_back(add_layer);
                layer_id++;


                int n = forward_layers.size();
                //downsample connected to last block
                forward_layers[n-3]->previous_layers.clear();
                forward_layers[n-3]->previous_layers.push_back(forward_layers[n-9]);
                forward_layers[n-9]->next_layers.push_back(forward_layers[n-3]);

                //downsample disconnected to bn2
                forward_layers[n-4]->next_layers.clear();

                //Connect Add layer
                forward_layers[n-4]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-4]);
                forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);

            }
            else  //It don't have downsample
            {
                //Create new "Add" layer
                Operatorr* add_op = new AddOp();
                Model_Layer* add_layer = new Model_Layer(add_op, layer_id);
                forward_layers.push_back(add_layer);
                layer_id++;

                int n = forward_layers.size();

                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-7]);
                forward_layers[n-7]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);

            }
        }
        else if (!strcmp(name->GetIdName(), "Bottleneck"))
        {
            if (layers->NumElements()==8)  //It has downsample
            {
                //Create new "Add" layer
                Operatorr* add_op = new AddOp();
                Model_Layer* add_layer = new Model_Layer(add_op, layer_id);
                forward_layers.push_back(add_layer);
                layer_id++;


                int n = forward_layers.size();
                //downsample connected to last block
                forward_layers[n-3]->previous_layers.clear();
                forward_layers[n-3]->previous_layers.push_back(forward_layers[n-11]);
                forward_layers[n-11]->next_layers.push_back(forward_layers[n-3]);

                //downsample disconnected to relu
                forward_layers[n-4]->next_layers.clear();

                //Connect Add layer
                forward_layers[n-4]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-4]);
                forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);

            }
            else
            {
                //Create new "Add" layer
                Operatorr* add_op = new AddOp();
                Model_Layer* add_layer = new Model_Layer(add_op, layer_id);
                forward_layers.push_back(add_layer);
                layer_id++;

                int n = forward_layers.size();

                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-9]);
                forward_layers[n-9]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);

            }
            
            
        }
    }
    if (is_inception)
    {
        if (!strcmp(name->GetIdName(), "InceptionA"))
        {
            yyltype random;
            int n = forward_layers.size();

            Operatorr* maxpl = new MaxPool2d(random, 3, 1, 1, 1, false);
            Model_Layer* maxpl_layer = new Model_Layer(maxpl, layer_id-3);
            forward_layers.insert(forward_layers.begin()+n-3, maxpl_layer);
            layer_id++;
            forward_layers[n-2]->layer_id = n-2;
            forward_layers[n-1]->layer_id = n-1;
            forward_layers[n]->layer_id = n;
            
            Operatorr* Concat_Op = new ConcatOp();
            Model_Layer* concat_layer = new Model_Layer(Concat_Op, layer_id);
            forward_layers.push_back(concat_layer);
            layer_id++;

            n = forward_layers.size();
            // cut
            forward_layers[n-21]->next_layers.clear();
            forward_layers[n-20]->previous_layers.clear();
            forward_layers[n-15]->next_layers.clear();
            forward_layers[n-14]->previous_layers.clear();
            forward_layers[n-6]->next_layers.clear();
            forward_layers[n-4]->previous_layers.clear();


            //connect
            forward_layers[n-21]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-15]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-6]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-21]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-15]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-6]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);

            forward_layers[n-24]->next_layers.push_back(forward_layers[n-20]);
            forward_layers[n-24]->next_layers.push_back(forward_layers[n-14]);
            forward_layers[n-24]->next_layers.push_back(forward_layers[n-5]);
            forward_layers[n-20]->previous_layers.push_back(forward_layers[n-24]);
            forward_layers[n-14]->previous_layers.push_back(forward_layers[n-24]);
            forward_layers[n-5]->previous_layers.push_back(forward_layers[n-24]);

            forward_layers[n-5]->next_layers.push_back(forward_layers[n-4]);
            forward_layers[n-4]->previous_layers.push_back(forward_layers[n-5]);
        }
        else if (!strcmp(name->GetIdName(), "InceptionB"))
        {
            yyltype random;
            Operatorr* maxpl = new MaxPool2d(random, 3, 2, 0, 1, false);
            Model_Layer* maxpl_layer = new Model_Layer(maxpl, layer_id);
            forward_layers.push_back(maxpl_layer);
            layer_id++;


            Operatorr* Concat_Op = new ConcatOp();
            Model_Layer* concat_layer = new Model_Layer(Concat_Op, layer_id);
            forward_layers.push_back(concat_layer);
            layer_id++;

            int n = forward_layers.size();
            // cut
            forward_layers[n-12]->next_layers.clear();
            forward_layers[n-11]->previous_layers.clear();


            //connect
            forward_layers[n-15]->next_layers.push_back(forward_layers[n-11]);
            forward_layers[n-15]->next_layers.push_back(forward_layers[n-2]);
            forward_layers[n-11]->previous_layers.push_back(forward_layers[n-15]);
            forward_layers[n-2]->previous_layers.push_back(forward_layers[n-15]);

            forward_layers[n-12]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-3]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-12]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-3]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);
        }
        else if (!strcmp(name->GetIdName(), "InceptionC"))
        {
            yyltype random;
            int n = forward_layers.size();

            Operatorr* maxpl = new MaxPool2d(random, 3, 1, 1, 1, false);
            Model_Layer* maxpl_layer = new Model_Layer(maxpl, layer_id-3);
            forward_layers.insert(forward_layers.begin()+n-3, maxpl_layer);
            layer_id++;
            forward_layers[n-2]->layer_id = n-2;
            forward_layers[n-1]->layer_id = n-1;
            forward_layers[n]->layer_id = n;
            
            Operatorr* Concat_Op = new ConcatOp();
            Model_Layer* concat_layer = new Model_Layer(Concat_Op, layer_id);
            forward_layers.push_back(concat_layer);
            layer_id++;

            n = forward_layers.size();

            // cut
            forward_layers[n-30]->next_layers.clear();
            forward_layers[n-29]->previous_layers.clear();
            forward_layers[n-21]->next_layers.clear();
            forward_layers[n-20]->previous_layers.clear();
            forward_layers[n-6]->next_layers.clear();
            forward_layers[n-4]->previous_layers.clear();


            //connect
            forward_layers[n-30]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-21]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-6]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-30]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-21]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-6]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);

            forward_layers[n-33]->next_layers.push_back(forward_layers[n-29]);
            forward_layers[n-33]->next_layers.push_back(forward_layers[n-20]);
            forward_layers[n-33]->next_layers.push_back(forward_layers[n-5]);
            forward_layers[n-29]->previous_layers.push_back(forward_layers[n-33]);
            forward_layers[n-20]->previous_layers.push_back(forward_layers[n-33]);
            forward_layers[n-5]->previous_layers.push_back(forward_layers[n-33]);

            forward_layers[n-5]->next_layers.push_back(forward_layers[n-4]);
            forward_layers[n-4]->previous_layers.push_back(forward_layers[n-5]);
        }
        else if (!strcmp(name->GetIdName(), "InceptionD"))
        {
            yyltype random;
            Operatorr* maxpl = new MaxPool2d(random, 3, 2, 0, 1, false);
            Model_Layer* maxpl_layer = new Model_Layer(maxpl, layer_id);
            forward_layers.push_back(maxpl_layer);
            layer_id++;


            Operatorr* Concat_Op = new ConcatOp();
            Model_Layer* concat_layer = new Model_Layer(Concat_Op, layer_id);
            forward_layers.push_back(concat_layer);
            layer_id++;

            int n = forward_layers.size();
            // cut
            forward_layers[n-15]->next_layers.clear();
            forward_layers[n-14]->previous_layers.clear();


            //connect
            forward_layers[n-21]->next_layers.push_back(forward_layers[n-14]);
            forward_layers[n-21]->next_layers.push_back(forward_layers[n-2]);
            forward_layers[n-14]->previous_layers.push_back(forward_layers[n-21]);
            forward_layers[n-2]->previous_layers.push_back(forward_layers[n-21]);

            forward_layers[n-15]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-3]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-15]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-3]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);
        }
        else if (!strcmp(name->GetIdName(), "InceptionE"))
        {
            yyltype random;
            int n = forward_layers.size();

            Operatorr* ConcatOp0 = new ConcatOp();
            Model_Layer* c0 = new Model_Layer(ConcatOp0, layer_id-15);
            layer_id++;
            forward_layers.insert(forward_layers.begin() + n - 15, c0);

            n++;

            Operatorr* ConcatOp1 = new ConcatOp();
            Model_Layer* c1 = new Model_Layer(ConcatOp1, layer_id-3);
            layer_id++;
            forward_layers.insert(forward_layers.begin() + n - 3, c1);

            n++;

            Operatorr* maxpl = new MaxPool2d(random, 3, 1, 1, 1, false);
            Model_Layer* maxpl_layer = new Model_Layer(maxpl, layer_id-3);
            forward_layers.insert(forward_layers.begin()+n-3, maxpl_layer);
            layer_id++;
            forward_layers[n-2]->layer_id = n-2;
            forward_layers[n-1]->layer_id = n-1;
            forward_layers[n]->layer_id = n;
            
            Operatorr* Concat_Op = new ConcatOp();
            Model_Layer* concat_layer = new Model_Layer(Concat_Op, layer_id);
            forward_layers.push_back(concat_layer);
            layer_id++;

            n = forward_layers.size();
            // cut
            forward_layers[n-29]->next_layers.clear();
            forward_layers[n-28]->previous_layers.clear();
            forward_layers[n-23]->next_layers.clear();
            forward_layers[n-22]->previous_layers.clear();
            forward_layers[n-20]->next_layers.clear();
            forward_layers[n-18]->previous_layers.clear();
            forward_layers[n-10]->next_layers.clear();
            forward_layers[n-9]->previous_layers.clear();
            forward_layers[n-7]->next_layers.clear();
            forward_layers[n-4]->previous_layers.clear();


            //connect
            forward_layers[n-29]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-19]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-6]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-29]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-19]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-6]);
            forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);

            forward_layers[n-32]->next_layers.push_back(forward_layers[n-28]);
            forward_layers[n-32]->next_layers.push_back(forward_layers[n-18]);
            forward_layers[n-32]->next_layers.push_back(forward_layers[n-5]);
            forward_layers[n-28]->previous_layers.push_back(forward_layers[n-32]);
            forward_layers[n-18]->previous_layers.push_back(forward_layers[n-32]);
            forward_layers[n-5]->previous_layers.push_back(forward_layers[n-32]);

            forward_layers[n-26]->next_layers.push_back(forward_layers[n-22]);
            forward_layers[n-22]->previous_layers.push_back(forward_layers[n-26]);
            forward_layers[n-23]->next_layers.push_back(forward_layers[n-19]);
            forward_layers[n-19]->previous_layers.push_back(forward_layers[n-23]);
            forward_layers[n-13]->next_layers.push_back(forward_layers[n-9]);
            forward_layers[n-9]->previous_layers.push_back(forward_layers[n-13]);
            forward_layers[n-10]->next_layers.push_back(forward_layers[n-6]);
            forward_layers[n-6]->previous_layers.push_back(forward_layers[n-10]);
            forward_layers[n-20]->next_layers.push_back(forward_layers[n-19]);
            forward_layers[n-19]->previous_layers.push_back(forward_layers[n-20]);
            forward_layers[n-7]->next_layers.push_back(forward_layers[n-6]);
            forward_layers[n-6]->previous_layers.push_back(forward_layers[n-7]);
            forward_layers[n-5]->next_layers.push_back(forward_layers[n-4]);
            forward_layers[n-4]->previous_layers.push_back(forward_layers[n-5]);
        }
        
    }
    else if (is_senet)
    {
        if (!strcmp(name->GetIdName(), "SENetUnit"))
        {
            if (layers->NumElements()==3) //Do not need to resize
            {
                int n = forward_layers.size();

                Operatorr* add_op = new AddOp();
                Model_Layer* c0 = new Model_Layer(add_op, layer_id - 1);
                layer_id++;
                forward_layers.insert(forward_layers.begin() + n - 1, c0);
                n++;
                forward_layers[n-1]->layer_id = n-1;

                forward_layers[n-3]->next_layers.clear();
                forward_layers[n-1]->previous_layers.clear();

                forward_layers[n-3]->next_layers.push_back(forward_layers[n-2]);
                forward_layers[n-2]->previous_layers.push_back(forward_layers[n-3]);
                forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);
                forward_layers[n-16]->next_layers.push_back(forward_layers[n-2]);
                forward_layers[n-2]->previous_layers.push_back(forward_layers[n-16]);

                
                n = forward_layers.size();

                Operatorr* scale_op = new ScaleOp();
                Model_Layer* scale_layer = new Model_Layer(scale_op, layer_id-2);
                forward_layers.insert(forward_layers.begin()+n-2, scale_layer);
                layer_id++;
                n++;
                forward_layers[n-2]->layer_id = n-2;
                forward_layers[n-1]->layer_id = n-1;


                forward_layers[n-4]->next_layers.clear();
                forward_layers[n-2]->previous_layers.clear();
                forward_layers[n-4]->next_layers.push_back(forward_layers[n-3]);
                forward_layers[n-3]->previous_layers.push_back(forward_layers[n-4]);
                forward_layers[n-3]->next_layers.push_back(forward_layers[n-2]);
                forward_layers[n-2]->previous_layers.push_back(forward_layers[n-3]);
                forward_layers[n-2]->previous_layers.push_back(forward_layers[n-17]);
                Assert(forward_layers[n-3]->operatorr->type==Scale_T);
                forward_layers[n-3]->get_scale_num_layer = forward_layers[n-17];

            }
            else if (layers->NumElements()==4)
            {
                int n = forward_layers.size();

                Operatorr* add_op = new AddOp();
                Model_Layer* c0 = new Model_Layer(add_op, layer_id - 1);
                layer_id++;
                forward_layers.insert(forward_layers.begin() + n - 1, c0);
                n++;
                forward_layers[n-1]->layer_id = n-1;

                forward_layers[n-3]->next_layers.clear();
                forward_layers[n-1]->previous_layers.clear();
                forward_layers[n-5]->next_layers.clear();
                forward_layers[n-4]->previous_layers.clear();

                forward_layers[n-5]->next_layers.push_back(forward_layers[n-2]);
                forward_layers[n-2]->previous_layers.push_back(forward_layers[n-5]);
                forward_layers[n-2]->next_layers.push_back(forward_layers[n-1]);
                forward_layers[n-1]->previous_layers.push_back(forward_layers[n-2]);
                forward_layers[n-18]->next_layers.push_back(forward_layers[n-4]);
                forward_layers[n-4]->previous_layers.push_back(forward_layers[n-18]);
                forward_layers[n-3]->next_layers.push_back(forward_layers[n-2]);
                forward_layers[n-2]->previous_layers.push_back(forward_layers[n-3]);


                n = forward_layers.size();

                Operatorr* scale_op = new ScaleOp();
                Model_Layer* scale_layer = new Model_Layer(scale_op, layer_id-4);
                forward_layers.insert(forward_layers.begin()+n-4, scale_layer);
                layer_id++;
                n++;
                forward_layers[n-4]->layer_id = n-4;
                forward_layers[n-3]->layer_id = n-3;
                forward_layers[n-2]->layer_id = n-2;
                forward_layers[n-1]->layer_id = n-1;


                forward_layers[n-6]->next_layers.clear();
                forward_layers[n-2]->previous_layers.clear();
                forward_layers[n-6]->next_layers.push_back(forward_layers[n-5]);
                forward_layers[n-5]->previous_layers.push_back(forward_layers[n-6]);
                forward_layers[n-5]->next_layers.push_back(forward_layers[n-2]);
                forward_layers[n-2]->previous_layers.push_back(forward_layers[n-5]);
                forward_layers[n-2]->previous_layers.push_back(forward_layers[n-3]);
                Assert(forward_layers[n-5]->operatorr->type==Scale_T);
                forward_layers[n-5]->get_scale_num_layer = forward_layers[n-3];
            }
            else
            {
                Assert(0);
            }
        }
        
            
    }
    
    
    
    
}


BlockLayer::BlockLayer(Identifier* id, Block* b) : Layer(id){
    Assert(id != NULL && b != NULL);
    (this->block=b)->SetParent(this);
}

void BlockLayer::PrintChildren(int indentLevel){
    id->Print(indentLevel+1, "Layername");
    block->Print(indentLevel+1, "Block");
}

void BlockLayer::Analysis(){
    block->Analysis();
}


SeqBlockLayer::SeqBlockLayer(Identifier *id, List<SeqBlock*> *seqblocks) : Layer(id){
    Assert(id != NULL && seqblocks != NULL);
    (this->seqblocks = seqblocks)->SetParentAll(this);
}

void SeqBlockLayer::PrintChildren(int indentLevel){
    id->Print(indentLevel+1, "Layername");
    seqblocks->PrintAll(indentLevel+1, "SeqBlocks");
}

void SeqBlockLayer::Analysis(){
    for (int i = 0; i < seqblocks->NumElements(); i++)
    {
        seqblocks->Nth(i)->Analysis();
    }
}



Model::Model(Identifier* name, List<Layer*> *layers){
    Assert(name != NULL && layers != NULL);
    (this->modelname = name)->SetParent(this);    
    (this->layers = layers)->SetParentAll(this);
}

void Model::PrintChildren(int indentLevel){
    Assert(layers !=NULL);
    modelname->Print(indentLevel+1, "Modelname");
    layers->PrintAll(indentLevel+1, "Layers");
}

void Model::Analysis(){
    for (int i = 0; i < layers->NumElements(); i++)
    {
        layers->Nth(i)->Analysis();
    }
}


void transformer_parse(std::string filename){
    std::ifstream fin(filename);
    if (!fin.good()) {
            printf("Invalid input NN model specified in config file <%s>\n", 
                    filename.c_str());
            Assert(false);
    }
    int op_id;
    int count = 0;
    while (fin>>op_id)
    {
        Model_OP* new_op = new Model_OP;
        new_op->op_id = op_id;
        fin>>new_op->type>>new_op->input_num;
        new_op->input_tensors.resize(new_op->input_num);

        for (int i = 0; i < new_op->input_num; i++)
        {
            fin>>new_op->input_tensors[i].tensor_id;
        }
        
        for (int i = 0; i < new_op->input_num; i++)
        {
            fin>>new_op->input_tensors[i].dim;
            new_op->input_tensors[i].dims.resize(new_op->input_tensors[i].dim);
            for (int j = 0; j < new_op->input_tensors[i].dim; j++)
            {
                fin>>new_op->input_tensors[i].dims[j];
                if (new_op->input_tensors[i].dims[j]==99)
                {
                    new_op->input_tensors[i].dims[j] = batch_size;
                }
            }
            transformer_tensors[new_op->input_tensors[i].tensor_id] = new_op->input_tensors[i];
        }
        std::string garbage;
        std::cout<<".";
        fin>>garbage;
        Assert(garbage=="-------------------");
        new_op->N = batch_size;

        // std::cout<<type_set<<std::endl;
        forward_ops.push_back(new_op);
        
        int replace;
        if (new_op->type=="SoftmaxBasic")
        {
            for (int j = 0; j < 3; j++)
            {
                fin>>op_id;
                Model_OP* new_opp = new Model_OP;
                new_opp->op_id = op_id;
                replace = op_id;
                fin>>new_opp->type>>new_opp->input_num;
                new_opp->input_tensors.resize(new_opp->input_num);

                for (int i = 0; i < new_opp->input_num; i++)
                {
                    fin>>new_opp->input_tensors[i].tensor_id;
                }
                
                for (int i = 0; i < new_opp->input_num; i++)
                {
                    fin>>new_opp->input_tensors[i].dim;
                    new_opp->input_tensors[i].dims.resize(new_opp->input_tensors[i].dim);
                    for (int j = 0; j < new_opp->input_tensors[i].dim; j++)
                    {
                        fin>>new_opp->input_tensors[i].dims[j];
                        if (new_opp->input_tensors[i].dims[j]==99)
                        {
                            new_opp->input_tensors[i].dims[j] = batch_size;
                        }
                    }
                    transformer_tensors[new_opp->input_tensors[i].tensor_id] = new_opp->input_tensors[i];
                }
                std::string garbage;
                fin>>garbage;
                Assert(garbage=="-------------------");
                delete new_opp;
            }
            new_op->op_id = replace;
        }
        op_map[new_op->op_id] = new_op;
        
    }
    if (borden>200 && batch_size ==512 && SSD_PCIe_bandwidth_GBps < 6.3)
    {
        loosen_parameter = 1.394;
    }
    else if(borden<200 && is_transformer==1 && batch_size ==1280 && SSD_PCIe_bandwidth_GBps > 6.3 && SSD_PCIe_bandwidth_GBps < 6.5){
        loosen_parameter = 1.1;
    }
    
}