#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple
import numpy as np
import ujson as json
import sys
import re

SHOULD_IGNORE_OP_TYPE = {
    # ignore these op types
    "Result": True,
    "Slice": True,
    "Reshape": True,
    "Broadcast": True,
    "Concat": True,
    "Convert": True,

    # do not ignore these op types
    "Dot": False,
    "Relu": False,
    "Add": False,
    "BatchMatMul": False,
    "Divide": False,
    "GatherV2": False,
    "Multiply": False,
    "Power": False,
    "SoftmaxBasic": False,
    "Sqrt": False,
    "Subtract": False,
    "Sum": False,
    "Tanh": False,
    "Convolution": False,
    "MaxPool": False,
    "Erf": False,
}

OP_TYPE_ELEM_ONE_INPUT = {
    "Relu",
    "Sqrt",
    "Tanh",
    "Erf",
}

OP_TYPE_ELEM_TWO_INPUTS = {
    "Add",
    "Divide",
    "Multiply",
    "Power",
    "Subtract",
}

OP_TYPE_REDUCE = {
    "Sum",
}

OP_TYPE_GATHER_V2 = {
    "GatherV2",
}

OP_TYPE_ID_REDUCE = 0
OP_TYPE_ID_RELU = 1
OP_TYPE_ID_ELEMENT = 2
OP_TYPE_ID_POOL = 3
OP_TYPE_ID_CONV = 4
OP_TYPE_ID_MATMUL = 5
OP_TYPE_ID_GATHER = 6

OP_TYPE_TO_TYPE_ID = {
    "Dot":          OP_TYPE_ID_MATMUL,
    "Relu":         OP_TYPE_ID_RELU,
    "Add":          OP_TYPE_ID_ELEMENT,
    "BatchMatMul":  OP_TYPE_ID_MATMUL,
    "Divide":       OP_TYPE_ID_ELEMENT,
    "GatherV2":     OP_TYPE_ID_GATHER,
    "Multiply":     OP_TYPE_ID_ELEMENT,
    "Power":        OP_TYPE_ID_ELEMENT,
    "SoftmaxBasic": OP_TYPE_ID_ELEMENT,
    "Sqrt":         OP_TYPE_ID_RELU,
    "Subtract":     OP_TYPE_ID_ELEMENT,
    "Sum":          OP_TYPE_ID_REDUCE,
    "Tanh":         OP_TYPE_ID_RELU,
    "Convolution":  OP_TYPE_ID_CONV,
    "MaxPool":      OP_TYPE_ID_POOL,
    "Erf":          OP_TYPE_ID_RELU,
}

class Operator:
    def __init__(self,
                 id: int,
                 ins_str: str,
                 name: str,
                 inputs: List[List[int]],
                 parse: bool = True):
        if not parse:
            return
        
        self.id: int = id
        self.ins_str: str = ins_str.strip()
        self.name: str = name
        
        self.parse_ins_str()

        self.inputs: List[int] = [v[0] for v in inputs]
        '''ids of operators where the inputs come from'''

        self.users: List[int] = []
        '''ids of operators that use this operator's output'''

        self.should_count_inputs: List[bool] = [True for i in self.inputs]

    def parse_ins_str(self):
        se = re.search("output.*\",\sinput_dict", self.ins_str)
        assert se, f"ins_str: {self.ins_str}"

        self.op_expression = self.ins_str[se.start():se.end()-13].strip()

        input_dict_str = self.ins_str[se.end()+1:]
        input_dict_str = input_dict_str[:input_dict_str.rfind(")")]
        self.input_dict: Dict[str, Dict[str, List[int]]] = json.loads(input_dict_str)

        for k, v in self.input_dict.items():
            if "dtype" in v:
                del v["dtype"]

        # print(f"op_expression: {self.op_expression}")
        # print(f"input_dict: {self.input_dict}")

    def dump_as_list(self) -> List:
        return [
            self.id,
            self.name,
            self.op_expression,
            self.input_dict,
            self.inputs,
            self.should_count_inputs,
        ]
    
    @staticmethod
    def from_list(l: List) -> "Operator":
        op = Operator(0, "", "", [], parse=False)
        op.id = l[0]
        op.name = l[1]
        op.op_expression = l[2]
        op.input_dict = l[3]
        op.inputs = l[4]
        op.should_count_inputs = l[5]
        return op


def parse_model(model_filename: str) -> List[Operator]:
    with open(model_filename, 'r') as f:
        model = json.load(f)

    operators = []
    for op in model:
        if op[2] == "Result":
            continue
        operators.append(Operator(op[0], op[1], op[2], op[3]))

    return operators

def get_dim_names_from_expr(op_expr: str, var_name: str) -> List[str]:
    # find "var_name[...] [=+-*/;]"
    se = re.search(f"{var_name}\[[BNMGSCKHOW\d,\s]*\]", op_expr)
    assert se, f"var_name; {var_name}, op_expr: {op_expr}"
    temp = se.group()
    assert temp.startswith(var_name), f"temp: {temp}"
    
    # find "[...]" for var_name
    se = re.search("\[.*\]", temp)
    assert se, temp
    var_dim_names = se.group().strip()[1:-1].split(",")
    var_dim_names = [v.strip() for v in var_dim_names]

    return var_dim_names

def get_dims_from_op_elem_two_inputs(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name in OP_TYPE_ELEM_TWO_INPUTS, f"op.name: {op.name}"
    assert len(op.input_dict) == 2, f"op.input_dict: {op.input_dict}"

    op_expr = op.op_expression
    output_dim_names = get_dim_names_from_expr(op_expr, "output0")
    input0_dim_names = get_dim_names_from_expr(op_expr, "input0")
    input1_dim_names = get_dim_names_from_expr(op_expr, "input1")

    assert output_dim_names == input0_dim_names == input1_dim_names, \
        f"\n\toutput_dim_names: {output_dim_names}, \n\tinput0_dim_names: {input0_dim_names}, \n\tinput1_dim_names: {input1_dim_names}"

    dim_lengths: List[int] = op.input_dict["input0"]["shape"]
    variables: List[List[List[int]]] = [
        [[i] for i in range(len(dim_lengths))],
        [[i] for i in range(len(dim_lengths))],
        [[i] for i in range(len(dim_lengths))],
    ]

    return dim_lengths, variables

def get_dims_from_op_elem_one_input(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name in OP_TYPE_ELEM_ONE_INPUT, f"op.name: {op.name}"
    assert len(op.input_dict) == 1, f"op.input_dict: {op.input_dict}"

    op_expr = op.op_expression
    output_dim_names = get_dim_names_from_expr(op_expr, "output0")
    input0_dim_names = get_dim_names_from_expr(op_expr, "input0")

    assert output_dim_names == input0_dim_names, \
        f"\n\toutput_dim_names: {output_dim_names}, \n\tinput0_dim_names: {input0_dim_names}"

    dim_lengths: List[int] = op.input_dict["input0"]["shape"]
    variables: List[List[List[int]]] = [
        [[i] for i in range(len(dim_lengths))],
        [[i] for i in range(len(dim_lengths))],
    ]

    return dim_lengths, variables

def get_dims_from_op_Dot(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "Dot", f"op.name: {op.name}"
    assert len(op.input_dict) == 2, f"op.input_dict: {op.input_dict}"

    op_expr = op.op_expression
    output_dim_names = get_dim_names_from_expr(op_expr, "output0")
    input0_dim_names = get_dim_names_from_expr(op_expr, "input0")
    input1_dim_names = get_dim_names_from_expr(op_expr, "input1")

    batch_dim_size = 1
    N = op.input_dict["input0"]["shape"][0]
    K = op.input_dict["input0"]["shape"][1]
    M = op.input_dict["input1"]["shape"][1]

    if len(output_dim_names) > 2:
        assert output_dim_names[0] == input0_dim_names[0] == "S0", \
            f"\n\toutput_dim_names: {output_dim_names}, \n\tinput0_dim_names: {input0_dim_names}, \n\tinput1_dim_names: {input1_dim_names}"
        batch_dim_size = op.input_dict["input0"]["shape"][0]
        output_dim_names = output_dim_names[1:]
        input0_dim_names = input0_dim_names[1:]
        N = op.input_dict["input0"]["shape"][1]
        K = op.input_dict["input0"]["shape"][2]

    transpose_input1 = output_dim_names[1] == input1_dim_names[0]

    # assert output_dim_names[0] == input0_dim_names[0] == "N" \
    #     and output_dim_names[1] == input1_dim_names[0] == "M" \
    #     and input0_dim_names[1] == input1_dim_names[0] == "K", \
    #     f"\n\toutput_dim_names: {output_dim_names}, \n\tinput0_dim_names: {input0_dim_names}, \n\tinput1_dim_names: {input1_dim_names}"

    dim_lengths: List[int] = [
        batch_dim_size * N, # S0*N
        K, # K
        M, # M
    ]

    variables: List[List[List[int]]] = [
        [[0], [2]],
        [[0], [1]],
        [[2], [1]] if transpose_input1 else [[1], [2]],
    ]

    return dim_lengths, variables

def get_dims_from_op_BatchMatMul(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "BatchMatMul", f"op.name: {op.name}"

    op_expr = op.op_expression
    output_dim_names = get_dim_names_from_expr(op_expr, "output0")
    input0_dim_names = get_dim_names_from_expr(op_expr, "input0")
    input1_dim_names = get_dim_names_from_expr(op_expr, "input1")
    assert output_dim_names[:2] == input0_dim_names[:2] == input1_dim_names[:2], \
        f"\n\toutput_dim_names: {output_dim_names}, \n\tinput0_dim_names: {input0_dim_names}, \n\tinput1_dim_names: {input1_dim_names}"

    assert output_dim_names[2] == input0_dim_names[2] == "N" \
        and output_dim_names[3] == input1_dim_names[3] == "M" \
        and input0_dim_names[3] == input1_dim_names[2] == "K", \
        f"\n\toutput_dim_names: {output_dim_names}, \n\tinput0_dim_names: {input0_dim_names}, \n\tinput1_dim_names: {input1_dim_names}"

    b0 = op.input_dict["input0"]["shape"][0]
    b1 = op.input_dict["input0"]["shape"][1]
    n = op.input_dict["input0"]["shape"][2]
    k = op.input_dict["input0"]["shape"][3]
    m = op.input_dict["input1"]["shape"][3]

    dim_lengths: List[int] = [b0, b1, n, k, m]
    variables: List[List[List[int]]] = [
        [[0], [1], [2], [4]],
        [[0], [1], [2], [3]],
        [[0], [1], [3], [4]],
    ]

    return dim_lengths, variables

def get_dims_from_op_Convolution(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "Convolution", f"op.name: {op.name}"

    op_expr = op.op_expression

    #################
    # !!! too lazy to do any useful check here, just hardcode everything
    #################

    dilated_factor = 1
    se_HO = re.search("(HO\s\*\s\d+)", op_expr)
    if se_HO:
        se_WO = re.search("(WO\s\*\s\d+)", op_expr)
        assert se_WO, f"se_HO: {se_HO}, se_WO: {se_WO}"
        dilated_factor = int(se_HO.group().split("*")[1].strip())

    dim_lengths: List[int] = [
        op.input_dict["input0"]["shape"][0], # batch (N)
        op.input_dict["input1"]["shape"][0], # out_chl (F)
        op.input_dict["input0"]["shape"][1], # input_chl (C)
        op.input_dict["input0"]["shape"][2] // dilated_factor, # out_hei (HO)
        op.input_dict["input0"]["shape"][3] // dilated_factor, # out_wid (WO)
        op.input_dict["input1"]["shape"][2], # ker_hei (KH)
        op.input_dict["input1"]["shape"][3], # ker_wid (KW)
    ]

    variables: List[List[List[int]]] = [ 
        [[0], [1], [3], [4]],
        [[0], [2], [3] * dilated_factor + [5], [4] * dilated_factor + [6]],
        [[2], [1], [5], [6]],
    ]

    return dim_lengths, variables

def get_dims_from_op_MaxPool(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "MaxPool", f"op.name: {op.name}"

    op_expr = op.op_expression

    #################
    # !!! too lazy to do any useful check here, just hardcode everything
    #################

    #  0:batches, 1:chl, 2:out_hei, 3:out_wid, 4:ker_hei, 5:ker_wid

    se_D0 = re.search("(D0\sin\s\d+\s*[,;])", op_expr)
    se_D1 = re.search("(D1\sin\s\d+\s*[,;])", op_expr)
    se_K0 = re.search("(K0\sin\s\d+\s*[,;])", op_expr)
    se_K1 = re.search("(K1\sin\s\d+\s*[,;])", op_expr)
    assert se_D0 and se_D1 and se_K0 and se_K1, f"se_D0: {se_D0}, se_D1: {se_D1}, se_K0: {se_K0}, se_K1: {se_K1}"

    D0 = int(re.split("[,;]", se_D0.group().split("in")[1].strip())[0].strip())
    D1 = int(re.split("[,;]", se_D1.group().split("in")[1].strip())[0].strip())
    K0 = int(re.split("[,;]", se_K0.group().split("in")[1].strip())[0].strip())
    K1 = int(re.split("[,;]", se_K1.group().split("in")[1].strip())[0].strip())

    pool_factor = op.input_dict["input0"]["shape"][2] // D0

    dim_lengths: List[int] = [
        op.input_dict["input0"]["shape"][0], # N
        op.input_dict["input0"]["shape"][1], # C
        D0, # D0
        D1, # D1
        K0, # K0
        K1, # K1
    ]

    variables: List[List[List[int]]] = [ 
        [[0], [1], [2],     [3]],
        [[0], [1], [2] * pool_factor + [4], [3] * pool_factor + [5]],
    ]

    return dim_lengths, variables

def get_dims_from_op_Reduce(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name in OP_TYPE_REDUCE, f"op.name: {op.name}"

    op_expr = op.op_expression

    output_dim_names = get_dim_names_from_expr(op_expr, "output0")
    input_dim_names = get_dim_names_from_expr(op_expr, "input0")

    assert len(output_dim_names) <= len(input_dim_names), \
        f"output_dim_names: {output_dim_names}, input_dim_names: {input_dim_names}"

    dim_lengths: List[int] = op.input_dict["input0"]["shape"]
    variables: List[List[List[int]]] = [
        [[i] for i in sorted([int(s[1]) for s in output_dim_names])],
        [[i] for i in sorted([int(s[1]) for s in input_dim_names])],
    ]

    return dim_lengths, variables

def get_dims_from_op_SoftmaxBasic(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "SoftmaxBasic", f"op.name: {op.name}"
    return get_dims_from_op_GatherV2(op)

def get_dims_from_op_GatherV2(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    # assert op.name == "GatherV2", f"op.name: {op.name}"

    op_expr = op.op_expression
    output_dim_names = get_dim_names_from_expr(op_expr, "output0")
    input0_dim_names = get_dim_names_from_expr(op_expr, "input0")
    input1_dim_names = []
    if len(op.input_dict) == 2:
        input1_dim_names = get_dim_names_from_expr(op_expr, "input1")

    variables = [[],[]]
    if len(op.input_dict) == 2:
        variables.append([])
    name_idx_dict = {}
    cur_idx = 0
    for out_name in output_dim_names:
        variables[0].append([cur_idx])
        name_idx_dict[out_name] = cur_idx
        cur_idx += 1
    
    if len(op.input_dict) == 2:
        for in1_name in input1_dim_names:
            if in1_name not in name_idx_dict:
                variables[2].append([cur_idx])
                name_idx_dict[in1_name] = cur_idx
                cur_idx += 1
            else:
                variables[2].append([name_idx_dict[in1_name]])

    for in0_name in input0_dim_names:
        if in0_name not in name_idx_dict:
            variables[1].append([cur_idx])
            name_idx_dict[in0_name] = cur_idx
            cur_idx += 1
        else:
            variables[1].append([name_idx_dict[in0_name]])

    dim_lengths = [0] * cur_idx
    if len(op.input_dict) == 2:
        for name, length in zip(input1_dim_names, op.input_dict["input1"]["shape"]):
            dim_lengths[name_idx_dict[name]] = length
    for name, length in zip(input0_dim_names, op.input_dict["input0"]["shape"]):
        dim_lengths[name_idx_dict[name]] = length

    return dim_lengths, variables

def get_dims_from_op(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    '''
    @returns a tuple of:
        dim_lengths: List[int],
        variables: List[List[List[int]]]
    '''
    if op.name in OP_TYPE_ELEM_ONE_INPUT:
        return get_dims_from_op_elem_one_input(op)
    elif op.name in OP_TYPE_ELEM_TWO_INPUTS:
        return get_dims_from_op_elem_two_inputs(op)
    elif op.name in OP_TYPE_REDUCE:
        return get_dims_from_op_Reduce(op)
    elif op.name in OP_TYPE_GATHER_V2:
        return get_dims_from_op_GatherV2(op)
    elif op.name == "Dot":
        return get_dims_from_op_Dot(op)
    elif op.name == "BatchMatMul":
        return get_dims_from_op_BatchMatMul(op)
    elif op.name == "Convolution":
        return get_dims_from_op_Convolution(op)
    elif op.name == "MaxPool":
        return get_dims_from_op_MaxPool(op)
    elif op.name == "SoftmaxBasic":
        return get_dims_from_op_SoftmaxBasic(op)
    elif op.name == "GatherV2":
        return get_dims_from_op_GatherV2(op)
    else:
        raise ValueError(f"op.name: {op.name}")

def get_tensor_expr_info_from_op(op: Operator) -> Tuple[str, List[int], List[List[List[int]]], List[bool], int]:
    '''
    @returns a tuple of:
        op_type_name: str,
        dim_lengths: List[int],
        variables: List[List[List[int]]],
        ignore_variables: List[bool],
        op_type_id: int,
    '''
    op_type_name = op.name
    assert not SHOULD_IGNORE_OP_TYPE[op_type_name], f"op_type_name: {op_type_name}"

    dim_lengths, variables = get_dims_from_op(op)

    ignore_variables: List[bool] = [not x for x in op.should_count_inputs]

    return op_type_name, dim_lengths, variables, ignore_variables, OP_TYPE_TO_TYPE_ID[op_type_name]

def load_ops_from_file(fname: str) -> List[Operator]:
    with open(fname, "r") as f:
        ops = json.load(f)
    return [Operator.from_list(op) for op in ops]

model_filename = sys.argv[1]
output_filename = f"parsed/parsed_{model_filename}"
texpre_filename = f"TExpr_{model_filename}"


if len(sys.argv) > 2:
    read_from_parsed_file = sys.argv[2] == "--parsed"
else:
    read_from_parsed_file = False

if read_from_parsed_file:
    with open(output_filename, 'r') as f:
        ops = load_ops_from_file(output_filename)
###########################################################
else:
    ops = parse_model(f"original/{model_filename}")

    ops_id_dict: Dict[int, Operator] = {op.id: op for op in ops}

    # find users of all ops
    for op in ops:
        for input_id in op.inputs:
            if input_id in ops_id_dict:
                ops_id_dict[input_id].users.append(op.id)

    # remove all ops that should be ignored
    # propagate their input_ids and uses to the remaining ops
    for op in ops:
        if not SHOULD_IGNORE_OP_TYPE[op.name]:
            continue

        # op.inputs should be added to op.users[x].inputs
        for user_id in op.users:
            if user_id in ops_id_dict:
                user = ops_id_dict[user_id]
                for input_id in op.inputs:
                    user.inputs.append(input_id)

        # op.users should be added to op.inputs[x].users
        for input_id in op.inputs:
            if input_id in ops_id_dict:
                input_op = ops_id_dict[input_id]
                for user_id in op.users:
                    input_op.users.append(user_id)

        # op.id should be removed from op.inputs[x].users and op.users[x].inputs
        for input_id in op.inputs:
            if input_id in ops_id_dict:
                input_op = ops_id_dict[input_id]
                assert op.id in input_op.users
                input_op.users.remove(op.id)
        for user_id in op.users:
            if user_id in ops_id_dict:
                user = ops_id_dict[user_id]
                assert op.id in user.inputs
                user.inputs.remove(op.id)

        # remove this op from ops_id_dict
        del ops_id_dict[op.id]

    # remove deleted ops from ops
    ops = [op for op in ops if op.id in ops_id_dict]

    for i in range(len(ops)-1):
        cur_op = ops[i]
        next_op = ops[i+1]
        if cur_op.id in next_op.inputs:
            if len(next_op.inputs) > len(next_op.should_count_inputs):
                pass # count all inputs for simplicity
            else:
                next_op.should_count_inputs[next_op.inputs.index(cur_op.id)] = False

for op in ops:
    print(op.id)
    print(op.name)
    #print(op.op_expression)
    print(op.input_dict.__len__())
    for num in op.inputs:
        print(num)
    for i in range(op.input_dict.__len__()):
        string1 = "input" + str(i)
        print(op.input_dict[string1]["shape"].__len__())
        for dim in op.input_dict[string1]["shape"]:
            print(dim)
    print("-------------------")

