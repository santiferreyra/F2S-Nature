"""Shared helpers for F2S training scripts.

This module holds everything that ``warm_start.py`` and ``cold_start.py`` have
in common: device setup, data loading, side-effect feature/graph preparation,
the model builder, and the save/eval routines (including the per-class and
macro F1 scores). The training loops themselves live in the scripts.
"""

import os

import numpy as np
import pandas as pd
import torch

import scipy.io as sio
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from chemprop.data import MoleculeDataset, MoleculeDatapoint
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from .modules import (
    F2S,
    MessagePassingEncoderBias,
    GATEncoderBias,
    SideEffectBertEmbeddingBias,
    SideEffectBertEmbeddingBiasDSGAT,
)
from .losses import DataDrivenLoss, DataDrivenLossWithL1


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(device)

DATA_FOLDER = "./data"
MODEL_FOLDER = "./saved_models"
NUM_SIDE_EFFECTS = 994
NUM_FREQ_CLASSES = 6  # frequency levels 0..5



# def read_smiles(file_path: str, column: str = "isomeric_smiles") -> list[str]:
#     """Read the isomeric SMILES column from a drug table."""
#     return pd.read_csv(file_path)[column].tolist()


def read_data(file_path: str) -> tuple[list[str], np.ndarray]:
    data = pd.read_csv(file_path)
    
    smiles_list = data['isomeric_smiles'].tolist()
    R_matrix = data.iloc[:, 3:].values
    
    return smiles_list, R_matrix


def load_side_effect_labels(file_path: str = None):
    """Load side-effect names and DSGAT node features from the ``.mat`` file."""
    if file_path is None:
        file_path = os.path.join(DATA_FOLDER, "side_effect_label.mat")
    mat_contents = sio.loadmat(file_path)
    se_names = [x[0][0] for x in mat_contents["side_effect"].tolist()]
    se_feats = mat_contents["node_label"]  # [num_side_effects, feat_dim]
    return se_names, se_feats


def extract_features(smiles_list: list[str]):
    """Featurise SMILES into chemprop molecule graphs."""
    molecules = [MoleculeDatapoint.from_smi(smi) for smi in smiles_list]
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    dataset = MoleculeDataset(molecules, featurizer)
    mol_graphs, *_ = zip(*dataset)

    return mol_graphs


def build_dataloader(mol_graphs, batch_size: int) -> DataLoader:
    """Build a PyG dataloader where each node carries source-atom + bond features."""
    datapoints = []
    for i, mol_graph in enumerate(mol_graphs):
        atom_datapoint = Data(
            x=torch.tensor(mol_graph.V, dtype=torch.float),
            edge_index=torch.tensor(mol_graph.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(mol_graph.E, dtype=torch.float),
            idx=torch.tensor(i, dtype=torch.long),
        )

        atom_datapoint.x = torch.cat(
            [atom_datapoint.x[atom_datapoint.edge_index[0, :]], atom_datapoint.edge_attr],
            dim=1,
        )

        datapoints.append(atom_datapoint)

    return DataLoader(
        datapoints,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
    )


def loader_from_smiles(smiles_list: list[str], batch_size: int) -> DataLoader:
    """Convenience: SMILES -> dataloader in one call."""
    return build_dataloader(extract_features(smiles_list), batch_size)



def load_bert_embeddings(se_names=None, pca_dim: int = 256):
    """Load PubMedBERT side-effect embeddings, optionally reordered to ``se_names``.

    The DSGAT scheme aligns the BERT rows to the ``.mat`` side-effect order
    (``reorder=True``); the plain BERT scheme uses the file order as-is, matching
    the original scripts. Returns the PCA-reduced embeddings and the retained
    variance ratio.
    """
    df = pd.read_csv(os.path.join(DATA_FOLDER, "bert", "side_effect_descriptions_cls_embeddings.csv"))
    bert_data = torch.tensor(df.values[:, :-1].astype(float)).float()


    pca = PCA(n_components=pca_dim)
    X_pca = pca.fit_transform(bert_data.cpu().numpy())
    bert_data = torch.tensor(X_pca).float()

    retained_var = float(pca.explained_variance_ratio_.sum())
    return bert_data, retained_var


def build_se_graph(R_train, se_feats, bert_data, se_feat: str = "both", k: int = 10) -> Data:
    """Build the side-effect kNN graph used by the DSGAT side-effect encoder.

    Edges come from a cosine kNN graph over the side-effect columns of
    ``R_train``. Node features are the DSGAT ``.mat`` features, the BERT
    embeddings, or their concatenation.

    NOTE: edges are indexed in R-column order while ``se_feats``/``bert_data``
    are in ``.mat`` order; this mirrors the original scripts and is preserved
    here unchanged.
    """
    A = kneighbors_graph(
        R_train.T.cpu().numpy(), k, mode="connectivity", metric="cosine", include_self=False
    )
    G = nx.from_numpy_array(A.todense())

    edges = []
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edges = torch.tensor(np.array(edges).T, dtype=torch.long).to(device)

    if se_feat == "dsgat":
        x_e = torch.from_numpy(se_feats).float().to(device)
    elif se_feat == "bert":
        x_e = bert_data.to(device)
    elif se_feat == "both":
        x_dsgat = torch.from_numpy(se_feats).float().to(device)
        x_bert = bert_data.to(device)
        x_e = torch.cat([x_dsgat, x_bert], dim=1)
    else:
        raise ValueError("se_feat must be 'dsgat', 'bert', or 'both'")

    return Data(x=x_e, edge_index=edges, device=device)

def build_model(encoder: str, se_scheme: str, drug_input_dim: int, embedding_size: int,
                hidden_channels: int, bert_data, se_graph: Data = None, global_bias=None) -> F2S:
    """Assemble an :class:`F2S` model from the chosen encoder / side-effect scheme."""
    if encoder == "mpnn":
        molecule_embedding = MessagePassingEncoderBias(
            drug_input_dim, out_channels=embedding_size, hidden_channels=hidden_channels
        )
    elif encoder == "gat":
        molecule_embedding = GATEncoderBias(
            drug_input_dim, out_channels=embedding_size, hidden_channels=hidden_channels
        )
    else:
        raise ValueError("encoder must be 'mpnn' or 'gat'")

    if se_scheme == "bert":
        side_effect_embedding = SideEffectBertEmbeddingBias(bert_data, embedding_size=embedding_size)
    elif se_scheme == "dsgat":
        if se_graph is None:
            raise ValueError("se_graph is required when se_scheme == 'dsgat'")
        side_effect_embedding = SideEffectBertEmbeddingBiasDSGAT(bert_data, se_graph, embedding_size=embedding_size)
    else:
        raise ValueError("se_scheme must be 'bert' or 'dsgat'")

    return F2S(molecule_embedding, side_effect_embedding, global_bias=global_bias)


def make_optimizer(model, optimizer: str, learning_rate: float, weight_decay: float):
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError("optimizer must be 'adamw' or 'adam'")


def predict(model, loader, R, send_embs: bool = False):
    """Run the model over ``loader`` and scatter predictions into a dense matrix."""
    preds = torch.zeros((R.shape[0], R.shape[1]), dtype=torch.float, device=device)

    if send_embs:
        mol_embs = torch.zeros((R.shape[0], model.molecule_embedding.out_channels), dtype=torch.float, device=device)
        side_embs = torch.zeros((R.shape[1], model.molecule_embedding.out_channels), dtype=torch.float, device=device)

    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            if send_embs:
                y_preds, idx, mol_embed, side_embed = model(batch_data, send_embs=True)
                mol_embs[idx, :] = mol_embed
                side_embs[:, :] = side_embed
            else:
                y_preds, idx = model(batch_data)

            preds[idx, :] = y_preds

    if send_embs:
        return preds, mol_embs, side_embs
    return preds



def _ensure_folder(name: str) -> str:
    folder_path = os.path.join(MODEL_FOLDER, name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_model(name: str, suffix: str, model: torch.nn.Module):
    folder_path = _ensure_folder(name)
    model_path = os.path.join(folder_path, "model" + suffix)
    torch.save(model.state_dict(), model_path + "_weights.pt")
    torch.save(model, model_path + "_obj.pt")
    print(f"Model saved to {folder_path}")


def save_losses(name: str, train_losses: list[float], val_losses: list[float], last_n: int = 500):
    folder_path = _ensure_folder(name)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, "loss_plot.png"))
    plt.close()

    # Zoom in on the tail of training
    if len(train_losses) > last_n:
        start = len(train_losses) - last_n
        plt.figure(figsize=(10, 6))
        plt.plot(range(start, len(train_losses)), train_losses[-last_n:], label="Train Loss")
        plt.plot(range(start, len(val_losses)), val_losses[-last_n:], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss (Last {last_n} epochs)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder_path, f"loss_plot_last_{last_n}.png"))
        plt.close()


def save_preds(name: str, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               R_train, R_val, suffix: str = "_final"):
    """Compute predictions, plot their distributions per frequency class, and return them."""
    folder_path = _ensure_folder(name)

    train_preds = predict(model, train_loader, R_train)
    val_preds = predict(model, val_loader, R_val)

    train_preds_dists = {}
    val_preds_dists = {}
    for i in range(NUM_FREQ_CLASSES):
        train_preds_dists[i] = train_preds[R_train == i].cpu().numpy()
        val_preds_dists[i] = val_preds[R_val == i].cpu().numpy()

    for dists, title, fname in (
        (train_preds_dists, "Training Predictions Distribution", "train_preds_dist"),
        (val_preds_dists, "Validation Predictions Distribution", "val_preds_dist"),
    ):
        plt.figure(figsize=(10, 6))
        for i in range(NUM_FREQ_CLASSES):
            sns.kdeplot(dists[i], label=f"Side Effect Value: {i}", fill=True, alpha=0.5)
        plt.xlabel("Predicted Values")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(folder_path, fname + suffix + ".png"))
        plt.close()

    return train_preds, val_preds, train_preds_dists, val_preds_dists


def kde_thresholds(train_preds_dists) -> list[float]:
    """Find frequency-class decision thresholds where adjacent KDE curves cross."""
    thresholds = [0]
    for j in range(NUM_FREQ_CLASSES - 1):
        data1 = np.array(train_preds_dists[j])
        data2 = np.array(train_preds_dists[j + 1])

        kde1 = gaussian_kde(data1)
        kde2 = gaussian_kde(data2)

        x = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)
        diff = kde1(x) - kde2(x)
        sign_change_indices = np.where(np.diff(np.sign(diff)))[0]

        crossings = []
        for i in sign_change_indices:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = diff[i], diff[i + 1]
            crossings.append(x0 - y0 * (x1 - x0) / (y1 - y0))  # linear interpolation

        if len(crossings) == 1:
            thresholds.append(crossings[0])
        else:
            for candidate in crossings:
                if candidate > thresholds[-1]:
                    thresholds.append(candidate)
                    break

    thresholds.pop(0)
    return thresholds


def classify(x, thresholds) -> int:
    """Map a continuous prediction to a frequency class given sorted thresholds."""
    for i, t in enumerate(thresholds):
        if x < t:
            return i
    return len(thresholds)


def save_confusion_matrix(model_name, train_preds_dists, val_preds_dists, suffix="_final"):
    """Derive KDE thresholds, build a row-normalised confusion matrix, and plot it."""
    folder_path = _ensure_folder(model_name)

    thresholds = kde_thresholds(train_preds_dists)

    conf = np.zeros((NUM_FREQ_CLASSES, NUM_FREQ_CLASSES))
    for i in range(NUM_FREQ_CLASSES):
        y = np.array([classify(x, thresholds) for x in val_preds_dists[i]])
        for j in range(NUM_FREQ_CLASSES):
            conf[i, j] = y[y == j].shape[0]
        conf[i, :] /= conf[i, :].sum()

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf * 100, annot=True, fmt=".2f")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.title("Confusion Matrix using KDE-based Thresholds")
    plt.savefig(os.path.join(folder_path, "confusion_matrix" + suffix + ".png"))
    plt.close()

    return thresholds, conf


def save_biases(model_name, model, train_loader, R_for_corr):
    """Extract molecule/side-effect biases, save them, and return their correlations.

    Returns ``(mol_corr, se_corr)`` where the correlations relate each bias to
    the popularity (number of known associations) of the drug / side effect.
    """
    folder_path = _ensure_folder(model_name)

    model.eval()
    mol_bias, indices = [], []
    with torch.no_grad():
        for batch_data in train_loader:
            model(batch_data)
            mol_bias.extend(model.mol_bias.squeeze(-1).cpu().numpy().tolist())
            indices.extend(idx_to_list(batch_data.idx))

    mol_bias_df = pd.DataFrame({"idx": indices, "bias": mol_bias}).set_index("idx").sort_index()
    mol_bias_df.to_csv(os.path.join(folder_path, "mol_bias.csv"))

    side_bias = model.side_bias.cpu().numpy()
    side_bias_df = pd.DataFrame({"idx": range(len(side_bias)), "bias": side_bias}).set_index("idx").sort_index()
    side_bias_df.to_csv(os.path.join(folder_path, "side_bias.csv"))

    return eval_correlations(mol_bias_df["bias"].values, side_bias_df["bias"].values, R_for_corr)


def idx_to_list(idx):
    return idx.cpu().numpy().tolist()


def save_signatures(model_name, model, smiles_list, R_full, batch_size):
    """Recompute embeddings/predictions over all drugs and dump them as tensors."""
    folder_path = _ensure_folder(model_name)

    loader = loader_from_smiles(smiles_list, batch_size)

    out_channels = model.molecule_embedding.out_channels
    R_preds = torch.zeros_like(R_full)
    mol_bias = torch.zeros((len(smiles_list), 1))
    W = torch.zeros((len(smiles_list), out_channels))
    H = torch.zeros((out_channels, R_full.shape[1]))

    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            y_pred, idx, mol_embed, side_embed = model(batch_data, send_embs=True)
            R_preds[idx, :] = y_pred
            mol_bias[idx, :] = model.mol_bias
            W[idx, :] = mol_embed
            H = side_embed

    torch.save(W, os.path.join(folder_path, "W.pt"))
    torch.save(H, os.path.join(folder_path, "H.pt"))
    torch.save(R_preds, os.path.join(folder_path, "R_preds.pt"))
    torch.save(model.side_bias, os.path.join(folder_path, "side_bias.pt"))
    torch.save(mol_bias, os.path.join(folder_path, "mol_bias.pt"))


def save_specs(name: str, specs: dict):
    folder_path = _ensure_folder(name)
    df = pd.DataFrame.from_dict(specs, orient="index", columns=["value"])
    df.index.name = "specification"
    df.to_csv(os.path.join(folder_path, "specs.csv"))



def eval_rmse(preds, R_val) -> float:
    squared_error = (preds - R_val) ** 2
    masked_error = squared_error[R_val > 0]
    return torch.sqrt(masked_error.mean()).item()


def eval_auroc(preds, R_train, R_val, is_cold_start=False) -> float:
    return _mean_ranking_metric(roc_auc_score, preds, R_train, R_val, is_cold_start)


def eval_auprc(preds, R_train, R_val, is_cold_start=False) -> float:
    return _mean_ranking_metric(average_precision_score, preds, R_train, R_val, is_cold_start)


def _mean_ranking_metric(metric_fn, preds, R_train, R_val, is_cold_start) -> float:
    """Average a ranking metric over drugs.

    Warm start scores only the side effects unobserved during training
    (``R_train == 0``); cold start scores every side effect.
    """
    scores = []
    for i in range(R_val.shape[0]):
        if is_cold_start:
            col_mask = slice(None)
        else:
            col_mask = (R_train[i, :] == 0)

        target = R_val[i, col_mask] > 0
        if target.sum() == 0:  # nothing to rank
            continue

        output = preds[i, col_mask]
        score = metric_fn(target.cpu().numpy(), output.cpu().numpy())
        if not np.isnan(score):
            scores.append(score)

    return float(np.mean(scores)) if scores else float("nan")


def eval_correlations(mol_biases, se_biases, R):
    """Correlate biases with drug / side-effect popularity."""
    drug_popularity = (R > 0).sum(axis=1)
    se_popularity = (R > 0).sum(axis=0)

    mol_corr = np.corrcoef(mol_biases, drug_popularity)[0, 1]
    se_corr = np.corrcoef(se_biases, se_popularity)[0, 1]

    return float(mol_corr), float(se_corr)


def eval_f1(preds, R, thresholds):
    """Per-class and macro-averaged F1 over the 6 frequency classes.

    Predictions are discretised into frequency classes 0..5 using the KDE
    thresholds, true labels are the integer frequencies in ``R``. Returns
    ``(per_class_f1, macro_f1)`` with ``per_class_f1`` a list of length 6.
    Yields NaNs if thresholds are unavailable.
    """
    labels = list(range(NUM_FREQ_CLASSES))

    if thresholds is None or len(thresholds) < NUM_FREQ_CLASSES - 1:
        return [float("nan")] * NUM_FREQ_CLASSES, float("nan")

    y_true = R.cpu().numpy().astype(int).ravel()
    y_pred = np.array([classify(x, thresholds) for x in preds.cpu().numpy().ravel()])

    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    return per_class.tolist(), float(macro)


def f1_specs(per_class_f1, macro_f1) -> dict:
    """Flatten F1 results into specs.csv entries."""
    specs = {f"f1_class_{i}": per_class_f1[i] for i in range(NUM_FREQ_CLASSES)}
    specs["f1_macro"] = macro_f1
    return specs