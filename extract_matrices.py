import os
import r2pipe
import json
import pydot
import numpy as np
from tqdm import tqdm
import torch

KNOWN_INSTR = set(['ircall', 'lea', 'shl', 'ill', 'upush', 'nop', 'cmp', 'call', 'ret', 'jmp', 'pop', 'rjmp', 'irjmp', 'or', 'div', 'sar', 'add', 'acmp', 'null', 'xor', 'mul', 'rcall', 'sub', 'cmov', 'push', 'not', 'cjmp', 'load', 'mov', 'invalid', 'and', 'rpush', 'io', 'shr', 'store', 'rol', 'mjmp', 'trap', 'ujmp', 'ror', 'abs', 'sal', 'swi', 'cswi','cjmp','cret'])

"""
Creates the feature matrix where each row are the features of a basic block.
"""
def create_feature_matrix(func_json_str, file_name, function_name):
    try:
        func_json_list = json.loads(func_json_str)    
        func_json = func_json_list[0]
        feature_matrix = torch.zeros((len(func_json['blocks']), 8), dtype=torch.float)
        for i, block in enumerate(func_json['blocks']):
            for instr in block['ops']:
                feature_matrix[i,0] += 1
                if 'type' in instr:
                    if 'add' == instr['type'] or 'sub' == instr['type'] or 'mul' == instr['type'] or 'div' == instr['type'] or 'abs' == instr['type']:
                        feature_matrix[i,1] += 1
                    elif 'call' in instr['type']:
                        feature_matrix[i,2] += 1
                    elif 'mov' in instr['type']:
                        feature_matrix[i,3] += 1
                    elif 'push' in instr['type'] or 'pop' in instr['type']:
                        feature_matrix[i,4] += 1
                    elif instr['type'] == 'and' or instr['type'] == 'or' or instr['type'] == 'xor' or instr['type'] == 'not' or instr['type'] == 'rol' or instr['type'] == 'ror' or instr['type'] == 'shl' or instr['type'] == 'sal' or instr['type'] == 'shr':
                        feature_matrix[i,5] += 1
                    elif instr['type'] == 'trap' or instr['type'] == 'swi' or instr['type'] == 'cswi' or instr['type'] == 'ill':
                        feature_matrix[i,6] += 1
                    elif instr['type'] not in KNOWN_INSTR:
                        print("WARNING: Instruction type found which is not yet known: ", instr['type'])
                if 'disasm' in instr:
                    # treats hard coded addresses also as constants
                    if " 0x" in instr:
                        feature_matrix[i,7] += 1
        return feature_matrix
    except:
        print("No function json found in the given string", func_json_str, "for function", function_name, "in file", file_name)
        return None


"""
Prints the feature matrix with feature headers.
"""
def print_feature_matrix(matrix):
    headers = ["total instr", "aritmhetic", "call", "move", "stack", "logical", "numeric"]
    column_widths = [max(len(str(row[i])) for row in matrix) for i in range(len(headers))]
    headers_widths = [len(header) for header in headers]
    max_widths = [max(column_widths[i], headers_widths[i]) for i in range(len(headers))]
    header_row = ' | '.join(header.ljust(max_widths[i]) for i, header in enumerate(headers))
    print(header_row)
    print('-' * len(header_row))  

    for row in matrix:
        formatted_row = ' | '.join(str(row[i]).ljust(max_widths[i]) for i in range(len(headers)))
        print(formatted_row)


""" 
Creates the edge list from the given .dot file from radare2 
"""
def create_edge_list(cfgraph, function_name, file_name):
    dot_graph = pydot.graph_from_dot_data(cfgraph)
    if dot_graph is None or len(dot_graph) == 0:
        print("No graph found for function", function_name, "in file", file_name, "with content", cfgraph)
        return None
    dot_graph = dot_graph[0]
    # Create a mapping from node names to indices, -3 to skip the first 3 meta information nodes
    node_map = {node.get_name(): i - 3 for i, node in enumerate(dot_graph.get_nodes())}
    source_nodes = []
    destination_nodes = []
    for edge in dot_graph.get_edges():
        if edge.get_source() in node_map and edge.get_destination() in node_map:
            source_nodes.append(node_map[edge.get_source()])
            destination_nodes.append(node_map[edge.get_destination()])
    edge_list = torch.tensor([source_nodes, destination_nodes], dtype=torch.int32)
    return edge_list 


"""
Prints the edge list.
"""
def print_tensor(tensor):
    # Convert tensor to numpy array for printing
    numpy_tensor = tensor.numpy()
    tensor_str = np.array2string(numpy_tensor, separator=' | ', formatter={'float_kind':lambda x: "%.2f" % x})
    print(tensor_str)


"""
Extracts the Control flow graphs with attributes for each binary function from the given file.
Saves the graphs in the given output directory, creating one if it does not exist.
"""
def extract_attributed_graphs(input_file, output_directory):
    print("Processing file", input_file, "...")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(input_file):
        raise ValueError("Path to input file", input_file,"does not exist")
    
    r2 = r2pipe.open(input_file)
    try:
        r2.cmd('aaa')  
    except Exception as e:
        os.remove(input_file)
        print("Error while analyzing file", input_file, "with error", e)
        return

    functions = r2.cmdj('aflj')
    loading_bar = tqdm(total=len(functions), desc="Processing functions")
    tensors = {}
    for function in functions:
        func_name = function['name']
        safe_func_name = func_name.replace(" ", "_")
        # extracts the call graph in a .dot file
        cfgraph = r2.cmd(f'agfd @ {func_name}')
        edge_list = create_edge_list(cfgraph, func_name, input_file)
        # extracts the binary function information in a json file
        func_json_str = r2.cmd(f'agfj @ {func_name}')
        feature_matrix = create_feature_matrix(func_json_str, input_file, func_name)
        if edge_list is not None and feature_matrix is not None:
            tensors[safe_func_name] = {'edges': edge_list, 'features': feature_matrix}
        loading_bar.update(1)

    # Save all tensors to a single file in the output_directory => one file per executable
    loading_bar.close()
    if len(tensors) > 0:
        torch.save(tensors, f'{output_directory}/{os.path.basename(input_file)}_tensors.pt')
    print("file", input_file, "processed successfully")
    r2.quit()


def main():
    unpacked_binaries_dir = os.path.join(os.getcwd(), 'benign\\samples-111-1')
    output_dir = os.path.join(os.getcwd(), 'benign_tensors')
    os.makedirs(output_dir, exist_ok=True)

    files1 = [os.path.join(root, file) for root, _, files in os.walk(unpacked_binaries_dir) for file in files]   
    files = [file for file in files1 if not os.path.exists(os.path.join(output_dir, os.path.basename(file) + '_tensors.pt'))]
    print("number of files already processed is ", len(files1) - len(files))
    for file in files:
        if not ".log" in file:
            extract_attributed_graphs(file, output_dir)


if __name__ == "__main__":
    main()