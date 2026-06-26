import torch
import torch.optim as optim

import time
import numpy as np
import pandas as pd
import scipy.io as sio
import networkx as nx

from torch_geometric.data import Data
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from src import *

### SETUP ###
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(device)

### LOADING DATA
filepath     = 'data/side_effect_label.mat'
mat_contents = sio.loadmat(filepath)

se_names = [x[0][0] for x in mat_contents['side_effect'].tolist()]
se_feats = mat_contents['node_label']

### HYPERPARAMETERS ###
FILE_PATH     = 'data/drugSMILES.csv'
DATA_FOLDER   = 'data'
SE_FEAT       = 'both'  # 'dsgat', 'bert', or 'both'
BATCH_SIZE    = 32
EPOCHS        = 5000
LOSS_ALPHA    = 0.03
LOSS_DELTA    = 0.0001
WEIGHT_DECAY  = 1e-5
LEARNING_RATE = 1e-4

# Read
from src.utils import read_data

smiles_list, R_matrix = read_data(FILE_PATH)
len_dataset = len(smiles_list)

smiles_train = smiles_list
smiles_val   = smiles_list
smiles_test  = smiles_list

### WARM START

R_train = pd.read_csv(DATA_FOLDER + '/R_train.csv', header=None).values
R_val   = pd.read_csv(DATA_FOLDER + '/R_val.csv',   header=None).values
R_test  = pd.read_csv(DATA_FOLDER + '/R_test.csv',  header=None).values

R_train = torch.Tensor(R_train).to(device)
R_val   = torch.Tensor(R_val).to(device)
R_test  = torch.Tensor(R_test).to(device)

### Build loaders
from src.utils import extract_features, build_dataloader

mol_graphs   = extract_features(smiles_train)
train_loader = build_dataloader(mol_graphs, BATCH_SIZE)

mol_graphs = extract_features(smiles_val)
val_loader = build_dataloader(mol_graphs, BATCH_SIZE)

mol_graphs = extract_features(smiles_test)
test_loader = build_dataloader(mol_graphs, BATCH_SIZE)

bert_data  = torch.tensor(pd.read_csv(DATA_FOLDER + '/bert/side_effect_descriptions_cls_embeddings.csv').values[:, :-1].astype(float)).float()
bert_names = pd.read_csv(DATA_FOLDER + '/bert/side_effect_descriptions_cls_embeddings.csv').values[:, -1].tolist()

index_map = {se_name: idx for idx, se_name in enumerate(bert_names)}
bert_idx  = [index_map[name] for name in se_names]
bert_data = bert_data[bert_idx]

pca_dim   = 256
pca       = PCA(n_components=pca_dim)
X_pca     = pca.fit_transform(bert_data.cpu().numpy())
bert_data = torch.tensor(X_pca).float()

explained_var_ratios = pca.explained_variance_ratio_
total_retained_variance = explained_var_ratios.sum()
print(f"Total variance retained after PCA: {total_retained_variance*100:.4f}")

A = kneighbors_graph(R_train.T.cpu().numpy(),10,mode='connectivity', metric='cosine', include_self=False)
G = nx.from_numpy_array(A.todense())

edges = []
for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])

edges = np.array(edges).T
edges = torch.tensor(edges, dtype=torch.long).to(device)

if SE_FEAT == 'dsgat':
    x_e = torch.from_numpy(se_feats).float().to(device)
elif SE_FEAT == 'bert':
    x_e = bert_data.to(device)
elif SE_FEAT == 'both':
    x_dsgat = torch.from_numpy(se_feats).float().to(device)
    x_bert = bert_data.to(device)
    x_e = torch.cat([x_dsgat, x_bert], dim=1)
else:
    raise ValueError("SE_FEAT must be 'dsgat', 'bert', or 'both'")

data_e = Data(x=x_e, edge_index=edges, device=device)

### NATURE RESULTS 200

from src.modules import F2S, GATEncoderBias, SideEffectBertEmbeddingBiasDSGAT
from src.losses import DataDrivenLossWithL1
from src.utils import *

models = []
for i in range(5):
    models.append({
            'name': f'F2S_WARM_START_200_{i}_Nature_Corrected',
            'learning_rate': LEARNING_RATE,
            'weight_decay':  WEIGHT_DECAY,
            'model': F2S(
                GATEncoderBias(86, out_channels=50*(i+2), hidden_channels=64),
                SideEffectBertEmbeddingBiasDSGAT(bert_data, data_e, embedding_size=50*(i+2)),
                global_bias=R_train.mean())
        })

# EPOCHS = 5 # for debugging
# EPOCHS = 5000 

for model_specs in models:
    model_name = model_specs['name']
    model      = model_specs['model']
    model      = model.to(device)
    
    # Get number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Define loss/optimizer
    criterion = DataDrivenLossWithL1(alpha=LOSS_ALPHA, delta=LOSS_DELTA)
    optimizer = optim.Adam(model.parameters(),
                        lr = LEARNING_RATE,
                        weight_decay = WEIGHT_DECAY)
    
    print(f"\n\n\n===== Training model {model_name}... =====\n\n\n")
    
    train_losses = []
    val_losses   = []
    rmses        = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_train_loss = 0
        epoch_val_loss   = 0
        
        # Train loop
        model.train()
        for batch_data in train_loader:
            optimizer.zero_grad()

            # Forward pass
            y_pred, idx, mol_embed, side_embed = model(batch_data, send_embs=True)  # [num_drugs, num_side_effects]
            y_true = R_train[idx, :]                                # Calculate loss
            loss = criterion(y_pred, y_true, mol_embed, side_embed)
            epoch_train_loss += loss.item()
        
            # Backward pass
            loss.backward()
            optimizer.step()

        train_losses.append(epoch_train_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} -",
                    f"Train Loss: {epoch_train_loss} -")
    
    end_time      = time.time()
    time_per_epoch = (end_time - start_time) / EPOCHS

    save_model(model_name, '_final', model)  # Save final model and prediction distributions
    train_preds, val_preds, train_preds_dists, val_preds_dists = save_preds(model_name, model, train_loader, val_loader, R_train, R_val)
    
    thresholds = [0]
    try:
        thresholds, conf_matrix = save_confusion_matrix(model_name, train_preds_dists, val_preds_dists)
    except:
        pass

    model.eval()
    mol_bias = []
    indices  = []
    with torch.no_grad():
        for batch_data in train_loader:
                # Forward pass
                y_pred, idx = model(batch_data)  # [num_drugs, num_side_effects]
                mol_bias.extend(model.mol_bias.squeeze(-1).cpu().detach().numpy().tolist())
                indices.extend(idx.cpu().detach().numpy().tolist())
    
    mol_bias = pd.DataFrame({'idx': indices, 'bias': mol_bias}).set_index('idx').sort_index()
    mol_bias.to_csv(f"saved_models/{model_name}/mol_bias.csv")
    
    # side_bias    = model.side_bias.cpu().detach().numpy()
    # side_bias_df = pd.DataFrame({'idx': range(len(side_bias)), 'bias': side_bias}).set_index('idx').sort_index()
    # side_bias_df.to_csv(f"saved_models/{model_name}/side_bias.csv")

    # mol_corr, se_corr = eval_correlations(mol_bias['bias'].values, side_bias_df['bias'].values, R_train.cpu().numpy())

    # Save model specs
    specs = {
        'model_name': model_name,
        'loss_alpha': LOSS_ALPHA,
        "loss_delta": LOSS_DELTA,
        'num_params': num_params,
        'time_per_epoch': time_per_epoch,
        'rmse_final': eval_rmse(val_preds, R_val),
        'auroc_final_R': eval_auroc(val_preds, R_train, R_val),
        'auprc_final_R': eval_auprc(val_preds, R_train, R_val),
        # 'mol_corr': mol_corr,
        # 'se_corr': se_corr,
        'PCA_dim': pca_dim,
        'retained_var': total_retained_variance,
        'global_bias': model.global_bias.cpu().detach().numpy(),
        'thresholds': thresholds,
        'confusion_matrix': conf_matrix,
    }
    
    save_specs(model_name, specs)
    
    # R_train = pd.read_csv(DATA_FOLDER + '/R_train.csv', header=None).values
    # R_val   = pd.read_csv(DATA_FOLDER + '/R_val.csv',   header=None).values
    # R_test  = pd.read_csv(DATA_FOLDER + '/R_test.csv',  header=None).values

    # R_train = torch.Tensor(R_train).to(device)
    # R_val   = torch.Tensor(R_val).to(device)
    # R_test  = torch.Tensor(R_test).to(device)
    # R_train = R_train + R_val + R_test

    # # Build loaders
    # mol_graphs   = extract_features(smiles_train)
    # train_loader = build_dataloader(mol_graphs, BATCH_SIZE)

    # R_preds = torch.zeros_like(R_train)
    # b_d     = torch.zeros(((len(smiles_list), 1)))
    # W       = torch.zeros((len(smiles_list), model.molecule_embedding.out_channels))
    # H       = torch.zeros((model.molecule_embedding.out_channels, 994))
    
    # epoch_train_loss = 0
    # model.eval()
    # with torch.no_grad():
    #     for batch_data in train_loader:
    #         # Forward pass
    #         y_pred, idx, mol_embed, side_embed = model(batch_data, send_embs=True)  # [num_drugs, num_side_effects]
            
    #         # Calculate loss
    #         y_true = R_train[idx, :]
    #         R_preds[idx, :] = y_pred
    #         b_d[idx, :] = model.mol_bias

    #         loss = criterion(y_pred, y_true, mol_embed, side_embed)
    #         epoch_train_loss += loss.item()

    #         W[idx, :] = mol_embed
    #         H = side_embed
    

    # MODEL_FOLDER = '.\\saved_models'
    # folder_path = os.path.join(MODEL_FOLDER, model_name)
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    
    # torch.save(W, os.path.join(folder_path, 'W.pt'))
    # torch.save(H, os.path.join(folder_path, 'H.pt'))
    # torch.save(R_preds, os.path.join(folder_path, 'R_preds.pt'))
    # torch.save(model.side_bias, os.path.join(folder_path, 'side_bias.pt'))
    # torch.save(b_d, os.path.join(folder_path, 'mol_bias.pt'))

    print('\n'*3 + '='*5 + f" Model {model_name} training complete! " + '='*5 + '\n'*3)