import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, log_loss
import numpy as np
from sklearn.model_selection import ParameterGrid, KFold
import csv
import visualization


class GraphClassifier(nn.Module): 
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_conv_layers, num_linear_layers):
        super(GraphClassifier, self).__init__()
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_conv_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.conv_layers.append(GCNConv(hidden_dim, embedding_dim))
        
        # Create linear layers
        linear_layers = []
        for _ in range(num_linear_layers - 1):
            linear_layers.append(nn.Linear(embedding_dim if _ == 0 else hidden_dim, hidden_dim))
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Dropout(p=0.5))
        linear_layers.append(nn.Linear(hidden_dim if num_linear_layers > 1 else embedding_dim, 1))
        
        self.classifier = nn.Sequential(*linear_layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))
        embeddings = global_mean_pool(x, batch)  # Aggregate node embeddings to get graph embedding

        x = self.classifier(embeddings)  
        graph_probs = torch.sigmoid(x)  # sigmoid act. fn. -> probability of graph being malicious
        return graph_probs.squeeze(-1), embeddings


class MILModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_conv_layers, num_linear_layers):
        super(MILModel, self).__init__()
        self.graph_classifier = GraphClassifier(input_dim, hidden_dim, embedding_dim, num_conv_layers, num_linear_layers)
    
    def forward(self, bags, device):
        bag_probabilities = []
        all_embeddings = []
        all_probabilities = []
        for bag in bags:
            bag = bag.to(device)
            graph_probs, embeddings = self.graph_classifier(bag)
            bag_prob = torch.mean(graph_probs)   # graph probabilities -> bags probability
            bag_probabilities.append(bag_prob)
            all_probabilities.extend(graph_probs.cpu().detach().numpy())
            all_embeddings.append(embeddings.cpu().detach().numpy())
        
        bag_probabilities = torch.stack(bag_probabilities)
        return bag_probabilities.squeeze(-1), all_embeddings, all_probabilities
    

class BagDataset(Dataset):
    def __init__(self, root_dir, label, file_extension=None):
        self.root_dir = root_dir
        self.label = label
        self.file_extension = file_extension
        self.files = [f for f in os.listdir(root_dir) if (file_extension is None or file_extension in f.lower())]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        bag_data = torch.load(os.path.join(self.root_dir, self.files[idx]))      
        if len(bag_data) == 0:
            return None
        bag = []
        for function_data in bag_data.values():
            edge_index = function_data['edges'].long()
            x = function_data['features'].float()
            x.requires_grad = True
            data = Data(x=x, edge_index=edge_index)
            bag.append(data)
        bag = Batch.from_data_list(bag)    
        return bag, self.label


class ValidationDataset(Dataset):
    def __init__(self, file_path):
        self.root_dir = file_path
        self.functions_data = torch.load(file_path)
        self.functions = list(self.functions_data.keys())
    
    def __len__(self):
        return len(self.functions_data)
    
    def __getitem__(self, idx):
        function_data = self.functions_data[self.functions[idx]]
        edge_index = function_data['edges'].long()
        x = function_data['features'].float()
        label = function_data['label']
        data = Data(x=x, edge_index=edge_index)
        return data, label
        

def custom_collate_fn(batch):
    bags, labels = zip(*batch) 
    labels = torch.tensor(labels)  
    return bags, labels


def train(model, optimizer, train_loader, device):
    model.train()
    loading_bar = tqdm(total=len(train_loader), desc="Training")
    for bags, labels in train_loader:
        labels = labels.float().to(device)
        optimizer.zero_grad()
        out, _, _ = model(bags, device)
        loss = F.binary_cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        loading_bar.update(1)
    loading_bar.close()


def validate(model, val_loader, device, epoch, log_file):
    model.eval()
    all_labels = []
    all_function_probabilities = []
    all_probabilities = []
    all_embeddings = []
    loading_bar = tqdm(total=len(val_loader), desc="Validating")
    for bags, labels in val_loader:
        labels = labels.float().to(device)
        out, embeddings, probs = model(bags, device)
        all_function_probabilities.extend(probs)
        all_embeddings.extend(embeddings)
        all_labels.extend(labels.tolist())
        all_probabilities.extend(out.tolist())
        loading_bar.update(1)
    loading_bar.close()
    # Visualize embeddings in 2D
    # visualization.plot_embeddings_2D(all_embeddings, all_labels, all_function_probabilities, "images/tsne_with_decision_boundary_epoch" + str(epoch))

    fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)
    prevalence = np.mean(all_labels)
    accuracies = (tpr * prevalence) + ((1 - fpr) * (1 - prevalence))
    optimal_idx = np.argmax(accuracies)
    accuracy_with_optimal_threshold = accuracies[optimal_idx]
    optimal_threshold = thresholds[optimal_idx]
    predictions = [1 if prob > optimal_threshold else 0 for prob in all_probabilities]
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    conf_matrix = confusion_matrix(all_labels, predictions)
    print("conf_matrix", conf_matrix)
    auc_value = auc(fpr, tpr)
    loss = log_loss(all_labels, all_probabilities)
    # Log metrics
    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if file is empty
            writer.writerow(['Epoch', 'Accuracy', 'Optimal Threshold', 'AUC', 'Precision', 'Recall', 'F1', 'Loss'])
        writer.writerow([epoch, accuracy_with_optimal_threshold, optimal_threshold, auc_value, precision, recall, f1, loss])
    
    # Plot ROC curve
    # visualization.plot_roc_curve(fpr, tpr, auc_value, epoch, 'images/roc_curve' + str(epoch) + '.png')
    return accuracy_with_optimal_threshold, auc_value 


def create_dataset(benign_tensors_dir='benign_training_data', malware_tensors_dir='malware_training_data'):
    datasets = []
    benign_dirs = [os.path.join(benign_tensors_dir, dir_name) for dir_name in os.listdir(benign_tensors_dir) if os.path.isdir(os.path.join(benign_tensors_dir, dir_name))]
    malware_dirs = [os.path.join(malware_tensors_dir, dir_name) for dir_name in os.listdir(malware_tensors_dir) if os.path.isdir(os.path.join(malware_tensors_dir, dir_name))]
    root_dirs = benign_dirs + malware_dirs
    loading_bar = tqdm(total=len(root_dirs), desc="Creating datasets")
    mal_count, benign_count = 0, 0
    for root_dir in root_dirs:
        if "benign" in root_dir:
            dataset = BagDataset(root_dir, 0)
            benign_count += len(dataset)
        else:
            dataset = BagDataset(root_dir, 1, file_extension=".rbot")
            mal_count += len(dataset)
        datasets.append(dataset)
        loading_bar.update(1)
    loading_bar.close()
    full_dataset = ConcatDataset(datasets)
    print("Number of bags in full dataset:", len(full_dataset))
    print("Number of benign bags:", benign_count, "Number of malware bags:", mal_count, "class imbalance:", benign_count/mal_count)
    return full_dataset


def train_model(model, train_loader, val_loader, device, nr_epochs, file_save_model, optimizer, nr_no_improvements=5):
    best_accuracy = 0.0
    best_auc = 0.0
    no_improvement_count = 0
    for i in range(nr_epochs):
        print(f"EPOCH {i}")
        train(model, optimizer, train_loader, device)
        acc, auc = validate(model, val_loader, device, i, "results2/" + file_save_model + ".csv")
        if acc > best_accuracy or auc > best_auc:
            best_accuracy = max(acc, best_accuracy)
            best_auc = max(auc, best_auc)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= nr_no_improvements:
            print("No improvement in accuracy or auc. Stopping training.")
            break
    # torch.save(model.state_dict(), file_save_model + ".pt")
    return best_accuracy, best_auc


def main(batch_size, hidden_dim, embedding_dim, learning_rate, nr_epochs, train_val_data_ratio, file_save_model, num_conv_layers, num_linear_layers, nr_no_improvements, use_fold=False):
    print("Training model with parameters: batch_size:", batch_size, "hidden_dim:", hidden_dim, "embedding_dim:", embedding_dim, "learning_rate:", learning_rate, "num_conv_layers:", num_conv_layers, "num_linear_layers:", num_linear_layers)
    full_dataset = create_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU")
    model = MILModel(8, hidden_dim, embedding_dim, num_conv_layers, num_linear_layers)  # 8 features per basic block
    model = model.to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not use_fold:
        train_size = int(train_val_data_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        mean_accuracy, _ = train_model(model, train_loader, val_loader, device, nr_epochs, file_save_model, optimizer, nr_no_improvements)

    if use_fold:
        fold = 3
        print("Using k-fold cross validation")
        if "fold_"+str(fold)+"_"+file_save_model+".csv" in os.listdir("results"):
            print("Model already trained. Skipping training.")
            accuracies = []
            for i in range(1, fold+1):
                csv_file = "fold_"+str(i)+"_"+file_save_model+".csv"
                with open("results2/"+csv_file, 'r') as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    max_accuracy = max(float(row[1]) for row in reader)
                    accuracies.append(max_accuracy)
            mean_accuracy = np.mean(accuracies)
            return mean_accuracy
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold = 0
        accuracies = []
        for train_idx, val_idx in kf.split(full_dataset):
            fold += 1
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_subsampler, collate_fn=custom_collate_fn,drop_last=True)
            val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_subsampler, collate_fn=custom_collate_fn, drop_last=True)
            acc, _ = train_model(model, train_loader, val_loader, device, nr_epochs, "fold_"+str(fold)+"_"+file_save_model, optimizer, nr_no_improvements)
            accuracies.append(acc)
        mean_accuracy = np.mean(accuracies)
    # Log final result in overall summary
    with open("results/summary", 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if file is empty
            writer.writerow(['model_name', 'mean_accuracy'])
        writer.writerow([file_save_model, mean_accuracy])
    return float(mean_accuracy)

if __name__ == "__main__":
    batch_size = 54
    hidden_dim = 128
    embedding_dim = 40
    learning_rate = 0.0011
    num_conv_layers = 3
    num_linear_layers = 3
    file_save_model = f"model_b{batch_size}_h{hidden_dim}_e{embedding_dim}_lr{learning_rate}_c{num_conv_layers}_l{num_linear_layers}"
    nr_epochs = 30 # Max number of epochs to train
    nr_no_improvements = 3 # The number of epochs to wait for improvement before stopping the training
    use_fold = False # If true, uses 3-fold cross validation
    train_val_data_ratio = 0.8 # Ratio of data used for training if not using fold
    main(batch_size, hidden_dim, embedding_dim, learning_rate, nr_epochs, train_val_data_ratio, "80_" + file_save_model, num_conv_layers, num_linear_layers, nr_no_improvements, use_fold)
    main(batch_size, hidden_dim, embedding_dim, learning_rate, nr_epochs, train_val_data_ratio, "802_" + file_save_model, num_conv_layers, num_linear_layers, nr_no_improvements, use_fold)
    main(batch_size, hidden_dim, embedding_dim, learning_rate, nr_epochs, .7, "70_" + file_save_model, num_conv_layers, num_linear_layers, nr_no_improvements, use_fold)
    main(batch_size, hidden_dim, embedding_dim, learning_rate, nr_epochs, .7, "702_" + file_save_model, num_conv_layers, num_linear_layers, nr_no_improvements, use_fold)
