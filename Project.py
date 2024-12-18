import numpy as np
from sklearn.model_selection import train_test_split
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from GraphTsetlinMachine.graphs import Graphs
import pandas as pd
import argparse
from time import time


def default_args(**kwargs):
#parameters
    epochs = 25
    board_size = 11
    depth = 5
    max_included_literals = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=epochs, type=int)
    parser.add_argument("--board_size", default=board_size, type=int)
    parser.add_argument("--depth", default=depth, type=int)
    parser.add_argument("--hypervector_size", default=512, type=int)
    parser.add_argument("--hypervector_bits", default=2, type=int)
    parser.add_argument("--message_size", default=512, type=int)
    parser.add_argument("--message_bits", default=2, type=int)
    parser.add_argument('--double_hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max_included_literals", default=max_included_literals, type=int)

    args = parser.parse_args(args=[])

    # here we set the specific parameters based on the board size
    if args.board_size == 3:
        args.number_of_clauses = 200
        args.T = 400
        args.s = 1.2
    elif args.board_size == 5:
        args.number_of_clauses = 1000
        args.T = 800
        args.s = 1.0
    elif args.board_size == 7:
        args.number_of_clauses = 1500
        args.T = 2000
        args.s = 0.9
    elif args.board_size == 9:
        args.number_of_clauses = 3200
        args.T = 4000
        args.s = 0.8
    elif args.board_size == 11:
        args.number_of_clauses = 4500
        args.T = 6000
        args.s = 0.8

    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    return args


#helper function
def position_to_edge_id(pos, board_size):
    return pos[0] * board_size + pos[1]

#init args
args = default_args()


# loading data, switch data based on what to run
data = pd.read_csv("hex_game_data_complete.csv", dtype=str)
#data = pd.read_csv("hex_game_data_2_moves_before.csv", dtype=str)
#data = pd.read_csv("hex_game_data_5_moves_before.csv", dtype=str)
board_size = args.board_size
node_names = [f"{i}_{j}" for i in range(1, board_size + 1) for j in range(1, board_size + 1)]
X = data[node_names].values  
y = data["Winner"].values.astype(int)

# split data (90-10)
split_idx = int(len(data) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# check how many there is of each winner (label distribution) in the training and test set
unique, counts = np.unique(y, return_counts=True)
label_distribution_train = dict(zip(unique, counts))
total_train = sum(counts)
print(f"Label distribution in training set: {label_distribution_train}")
print(f"Percentage of '0's: {(label_distribution_train.get(0, 0) / total_train) * 100:.2f}%")
print(f"Percentage of '1's: {(label_distribution_train.get(1, 0) / total_train) * 100:.2f}%")

unique_test, counts_test = np.unique(y_test, return_counts=True)
label_distribution_test = dict(zip(unique_test, counts_test))
total_test = sum(counts_test)
print(f"Label distribution in test set: {label_distribution_test}")
print(f"Percentage of '0's: {(label_distribution_test.get(0, 0) / total_test) * 100:.2f}%")
print(f"Percentage of '1's: {(label_distribution_test.get(1, 0) / total_test) * 100:.2f}%")


# define edges for nodes
edges = []
for i in range(board_size):
    for j in range(board_size):
        if j < board_size - 1:
            edges.append(((i, j), (i, j + 1)))
        if i < board_size - 1:
            edges.append(((i, j), (i + 1, j)))
        if i < board_size - 1 and j > 0:
            edges.append(((i, j), (i + 1, j - 1)))
#limit for nodes edge based on position            
n_edges_list = []
for i in range(board_size ** 2):
    if i == 0 or i == board_size ** 2 - 1:
        n_edges_list.append(2)  # corners 2 neighbors
    elif i == board_size - 1 or i == board_size ** 2 - board_size:
        n_edges_list.append(3)  # other corners 3 neighbors
    elif i // board_size == 0 or i // board_size == board_size - 1:
        n_edges_list.append(4)  # top/bottom nodes
    elif i % board_size == 0 or i % board_size == board_size - 1:
        n_edges_list.append(4)  # left/right nodes
    else:
        n_edges_list.append(6)  # center nodes

#creating traning graph
graphs_train = Graphs(
    number_of_graphs=len(X_train),
    symbols=["O", "X", "."],
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing=args.double_hashing
)

for graph_id in range(X_train.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, board_size ** 2)

graphs_train.prepare_node_configuration()

# add nodes and properties to each graph
for graph_id, board in enumerate(X_train):
    for node_id, node_name in enumerate(node_names):  
        graphs_train.add_graph_node(graph_id, node_name, n_edges_list[node_id])
        sym = board[node_id]  
        graphs_train.add_graph_node_property(graph_id, node_name, sym)
graphs_train.prepare_edge_configuration()

# add edges to the graphs to represent connections between nodes
for graph_id in range(X_train.shape[0]):
    for edge in edges: 
        node_id = position_to_edge_id(edge[0], board_size)
        destination_node_id = position_to_edge_id(edge[1], board_size)
        graphs_train.add_graph_node_edge(graph_id, node_names[node_id], node_names[destination_node_id], edge_type_name="Plain")
        graphs_train.add_graph_node_edge(graph_id, node_names[destination_node_id], node_names[node_id], edge_type_name="Plain")

graphs_train.encode()


# create and encode testing graphs
graphs_test = Graphs(len(X_test), init_with=graphs_train)

for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, board_size ** 2)

graphs_test.prepare_node_configuration()

for graph_id, board in enumerate(X_test):
    for node_id, node_name in enumerate(node_names):
        graphs_test.add_graph_node(graph_id, node_name, n_edges_list[node_id])
        graphs_test.add_graph_node_property(graph_id, node_name, board[node_id])

graphs_test.prepare_edge_configuration()

for graph_id in range(X_test.shape[0]):
    for edge in edges:
        src_id = position_to_edge_id(edge[0], board_size)
        dest_id = position_to_edge_id(edge[1], board_size)
        graphs_test.add_graph_node_edge(graph_id, node_names[src_id], node_names[dest_id], edge_type_name="Plain")
        graphs_test.add_graph_node_edge(graph_id, node_names[dest_id], node_names[src_id], edge_type_name="Plain")

graphs_test.encode()


# train and evaluate the gtm
tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals,
    grid=(16 * 13, 1, 1),
    block=(128, 1, 1)
)

start_training = time()
#save accuracies
train_accuracies = []
test_accuracies = []
'''
#can print the best one
best_train_acc = -1  
best_test_acc = -1
best_train_predictions = None
best_test_predictions = None
'''
for i in range(args.epochs):
    if len(y_train) != graphs_train.number_of_graphs:
        raise ValueError(f"Mismatch: {len(y_train)} labels but {graphs_train.number_of_graphs} graphs.")

    tm.fit(graphs_train, y_train, epochs=1, incremental=True)
    train_acc = np.mean(y_train == tm.predict(graphs_train))
    test_acc = np.mean(y_test == tm.predict(graphs_test))
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    print(f"Epoch#{i + 1} -- Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

'''
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_train_acc = train_acc
        best_train_predictions = tm.predict(graphs_train)
        best_test_predictions = tm.predict(graphs_test)
''' 
print(f"Training time: {time() - start_training:.2f} seconds")
print(f"Training graphs: {graphs_train.number_of_graphs}, Nodes per graph: {board_size ** 2}")
print(f"Testing graphs: {graphs_test.number_of_graphs}, Nodes per graph: {board_size ** 2}")

# calculate average, best, and worst accuracies
average_train_acc = np.mean(train_accuracies)
average_test_acc = np.mean(test_accuracies)

best_train_acc = np.max(train_accuracies)
worst_train_acc = np.min(train_accuracies)

best_test_acc = np.max(test_accuracies)
worst_test_acc = np.min(test_accuracies)

# print average, best, and worst accuracies
print(f"\nAverage Train Accuracy: {average_train_acc * 100:.2f}%")
print(f"Average Test Accuracy: {average_test_acc * 100:.2f}%")
print(f"Best Train Accuracy: {best_train_acc * 100:.2f}%")
print(f"Worst Train Accuracy: {worst_train_acc * 100:.2f}%")
print(f"Best Test Accuracy: {best_test_acc * 100:.2f}%")
print(f"Worst Test Accuracy: {worst_test_acc * 100:.2f}%")

print(f"Training time: {time() - start_training:.2f} seconds")
'''
# save best final predictions to file
np.savetxt("best_train_predictions_results.txt", np.column_stack((best_train_predictions, y_train)), fmt="%d", header="Predicted\tActual")
np.savetxt("best_test_predictions_results.txt", np.column_stack((best_test_predictions, y_test)), fmt="%d", header="Predicted\tActual")
'''

def print_clause_weights(tm):
    weights = tm.get_state()[1].reshape(2, -1) 
    for i in range(tm.number_of_clauses):
        print(f"Clause #{i} - Weights: (Positive: {weights[0, i]}, Negative: {weights[1, i]})", end=" ")
        literals = []
        for k in range(tm.number_of_features * 2):
            if tm.ta_action(0, i, k):
                literals.append(f"x{k}" if k < tm.number_of_features else f"NOT x{k - tm.number_of_features}")
        print(" AND ".join(literals))

# print clause weights
#print_clause_weights(tm)

