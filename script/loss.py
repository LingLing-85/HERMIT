import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score
from script.config import args
# from script.utils.util import negative_sampling
from script.hgcn.manifolds import PoincareBall, Hyperboloid
from torch_geometric.utils import negative_sampling
from script.utils.util import logger
from torch.nn.modules.loss import BCEWithLogitsLoss

device = args.device

EPS = args.EPS
MAX_LOGVAR = 10


class RTTRegressionHead(nn.Module):
    """
    RTT Regression Head using concatenated node embeddings
    Predicts RTT values from [z_u || z_v]
    """
    def __init__(self, manifold, input_dim, hidden_dim=32, c=1.0):
        super(RTTRegressionHead, self).__init__()
        self.manifold = manifold
        self.c = c
        # Input is concatenation of two node embeddings: 2 * input_dim
        self.input_dim = input_dim * 2
        
        # MLP: [z_u || z_v] -> hidden -> RTT (log-normalized scale) -> [0, 1]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Add dropout for regularization
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Constrain output to [0, 1] for normalized RTT
        )
    
    def forward(self, z, edge_index):
        """
        Args:
            z: [N, D] node embeddings in hyperbolic space
            edge_index: [2, E] edge indices
        Returns:
            pred_rtt: [E] predicted RTT values (log-normalized scale)
        """
        z_i = torch.nn.functional.embedding(edge_index[0], z)
        z_j = torch.nn.functional.embedding(edge_index[1], z)
        
        # Concatenate embeddings
        combined = torch.cat([z_i, z_j], dim=-1) # [E, 2*D]
        
        # MLP prediction
        pred_rtt = self.mlp(combined).squeeze(-1)  # [E]
        return pred_rtt


class ReconLoss(nn.Module):
    def __init__(self, args, manifold=None):
        super(ReconLoss, self).__init__()
        self.negative_sampling = negative_sampling
        self.sampling_times = args.sampling_times
        self.r = 2.0
        self.t = 1.0
        self.sigmoid = True
        self.manifold = manifold if manifold else PoincareBall()
        self.use_hyperdecoder = args.use_hyperdecoder and args.model in ['HMPTGN']
        logger.info('using hyper decoder' if self.use_hyperdecoder else "not using hyper decoder")
        
        # RTT Prediction setup
        self.enable_rtt_prediction = args.enable_rtt_prediction
        if self.enable_rtt_prediction:
            self.rtt_head = RTTRegressionHead(
                self.manifold,
                input_dim=args.nout,  # Pass embedding dimension
                hidden_dim=args.rtt_hidden_dim,
                c=args.curvature
            ).to(args.device)  # Move RTT head to correct device
            self.rtt_loss_weight = args.rtt_loss_weight
            logger.info(f'RTT prediction enabled (weight={self.rtt_loss_weight}, hidden_dim={args.rtt_hidden_dim}, mode=Concat)')

    @staticmethod
    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def hyperdeoder(self, z, edge_index):
        def FermiDirac(dist):
            # Clamp dist to avoid overflow in exp
            dist = torch.clamp(dist, max=50.0)
            probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
            return probs

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        z_i = torch.nn.functional.embedding(edge_i, z)
        z_j = torch.nn.functional.embedding(edge_j, z)
        dist = self.manifold.sqdist(z_i, z_j, c=1.0)
        return FermiDirac(dist)

    def forward(self, z, pos_edge_index, neg_edge_index=None, rtt_labels=None):
        # Task A: Link Prediction Loss
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder
        
        pos_probs = decoder(z, pos_edge_index)
        pos_probs = torch.clamp(pos_probs, min=EPS, max=1-EPS)
        pos_loss = -torch.log(pos_probs).mean()
        
        if neg_edge_index == None:
            neg_edge_index = negative_sampling(pos_edge_index,
                                               num_neg_samples=pos_edge_index.size(1) * self.sampling_times)
        
        neg_probs = decoder(z, neg_edge_index)
        neg_probs = torch.clamp(neg_probs, min=EPS, max=1-EPS)
        neg_loss = -torch.log(1 - neg_probs).mean()
        link_loss = pos_loss + neg_loss
        
        # Task B: RTT Prediction Loss (only for positive edges)
        rtt_loss = 0.0
        if self.enable_rtt_prediction and rtt_labels is not None:
            pred_rtt = self.rtt_head(z, pos_edge_index)  # [num_pos_edges]
            # MSE Loss on log-normalized scale
            rtt_loss_raw = F.mse_loss(pred_rtt, rtt_labels)
            rtt_loss = self.rtt_loss_weight * rtt_loss_raw
            
            # Diagnostic logging (1% chance)
            if torch.rand(1).item() < 0.01:
                logger.info(f'DEBUG Loss: link={link_loss.item():.4f}, rtt_raw={rtt_loss_raw.item():.4f}, rtt_weighted={rtt_loss.item():.4f}')
        
        return link_loss + rtt_loss

    def predict(self, z, pos_edge_index, neg_edge_index):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder

        pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        pred_label = (pred > 0.5).astype(int)
        
        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        f1 = f1_score(y, pred_label)
        acc = accuracy_score(y, pred_label)
        prec = precision_score(y, pred_label)
        rec = recall_score(y, pred_label)
        
        return auc, ap, f1, acc, prec, rec
    
    def predict_rtt(self, z, edge_index, rtt_labels_normalized, avg_min, avg_max):
        """
        Evaluate RTT prediction with metrics in original millisecond scale
        
        Args:
            z: node embeddings
            edge_index: edge indices
            rtt_labels_normalized: log-normalized RTT labels [0, 1] avg_min, avg_max: normalization parameters for inverse transform
        Returns:
            mae, rmse in milliseconds
        """
        with torch.no_grad():
            pred_norm = self.rtt_head(z, edge_index)  # log-normalized scale [0, 1]
            # Step 1: Denormalize [0,1] -> log scale
            pred_log = pred_norm * (avg_max - avg_min) + avg_min
            true_log = rtt_labels_normalized * (avg_max - avg_min) + avg_min
            # Step 2: expm1 to restore original ms
            pred_ms = torch.expm1(pred_log)
            true_ms = torch.expm1(true_log)
            # Compute metrics
            mae = F.l1_loss(pred_ms, true_ms)
            rmse = torch.sqrt(F.mse_loss(pred_ms, true_ms))
        return mae.item(), rmse.item()


class VGAEloss(ReconLoss):
    def __init__(self, args):
        super(VGAEloss, self).__init__(args)

    def kl_loss(self, mu=None, logvar=None):
        mu = self.__mu__ if mu is None else mu
        logvar = self.__logvar__ if logvar is None else logvar.clamp(
            max=MAX_LOGVAR)
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    def forward(self, x, pos_edge_index, neg_edge_index):
        z, mu, logvar = x
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        reconloss = pos_loss + neg_loss
        klloss = (1 / z.size(0)) * self.kl_loss(mu=mu, logvar=logvar)

        return reconloss + klloss

class DySATloss(nn.Module):
    def __init__(self, args):
        super(DySATloss, self).__init__()
        self.bceloss = BCEWithLogitsLoss()
        if args.dysat_window < 0:
            self.num_time_steps = args.train_length
        else:
            self.num_time_steps = min(args.train_length, args.dysat_window + 1)  # window = 0 => only self.
    
    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, z, list_pos_edge_index, list_neg_edge_index):
        graph_loss = 0
        for t in range(len(list_pos_edge_index)):
            emb_t = z[:, t, :].squeeze() #[N, F]
            pos_source_node_emb = emb_t[list_pos_edge_index[t][0]] #[N', F]
            pos_tar_node_emb = emb_t[list_pos_edge_index[t][1]]    #[N', F]
            neg_source_node_emb = emb_t[list_neg_edge_index[t][0]] #[N', F]
            neg_tar_node_emb = emb_t[list_neg_edge_index[t][1]]    #[N', F]
            pos_score = torch.sum(pos_source_node_emb*pos_tar_node_emb, dim=1)
            neg_score = -torch.sum(neg_source_node_emb*neg_tar_node_emb, dim=1)
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))
            graphloss = pos_loss + neg_loss
            graph_loss += graphloss
        return graph_loss

    def predict(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = self.decoder(z, pos_edge_index)
        neg_pred = self.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)


class DHGATloss(nn.Module):
    def __init__(self, args):
        super(DHGATloss, self).__init__()
        self.negative_sampling = negative_sampling
        self.sampling_times = args.sampling_times
        self.r = 2.0
        self.t = 1.0
        self.sigmoid = True
        self.manifold = Hyperboloid()
        self.use_hyperdecoder = args.use_hyperdecoder and args.model in ['DHGAT']
        logger.info('using hyper decoder' if self.use_hyperdecoder else "not using hyper decoder")

    @staticmethod
    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def hyperdeoder(self, z, edge_index):
        def FermiDirac(dist):
            probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
            return probs

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        z_i = torch.nn.functional.embedding(edge_i, z)
        z_j = torch.nn.functional.embedding(edge_j, z)
        dist = self.manifold.sqdist(z_i, z_j, c=1.0)
        return FermiDirac(dist)

    def forward(self, z, pos_edge_index, neg_edge_index=None):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder
        pos_loss = -torch.log(
            decoder(z, pos_edge_index) + EPS).mean()
        if neg_edge_index == None:
            neg_edge_index = negative_sampling(pos_edge_index,
                                               num_neg_samples=pos_edge_index.size(1) * self.sampling_times)
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder

        pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)
