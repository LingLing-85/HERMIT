import os
import sys
import time
import torch
import numpy as np
from math import isnan

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

#torch.autograd.set_detect_anomaly(True)

class Runner(object):
    def __init__(self):
        self.len = data['time_length']
        self.start_train = 0
        
        # Calculate split points
        # Total = Train + Val + Test
        # Test is fixed by args.testlength
        # Val is fixed by args.vallength (if > 0)
        # Remainder is Train
        self.test_len = args.testlength
        self.val_len = getattr(args, 'vallength', 0)
        self.train_len = self.len - self.test_len - self.val_len
        
        if self.train_len <= 0:
             raise ValueError(f"Training length non-positive: {self.len} - {self.test_len} - {self.val_len}")
             
        self.train_shots = list(range(0, self.train_len))
        self.val_shots = list(range(self.train_len, self.train_len + self.val_len))
        self.test_shots = list(range(self.train_len + self.val_len, self.len))
        
        logger.info(f'Split: Train={len(self.train_shots)}, Val={len(self.val_shots)}, Test={len(self.test_shots)}')

        self.load_feature()
        self.model = load_model(args).to(args.device)
        self.model_name = args.model
        # Pass manifold to ReconLoss for RTT prediction
        if args.model == 'HMPTGN':
            self.loss = ReconLoss(args, manifold=self.model.manifold) if args.model not in ['DynVAE', 'VGRNN', 'HVGRNN'] else VGAEloss(args)
        else:
            self.loss = ReconLoss(args) if args.model not in ['DynVAE', 'VGRNN', 'HVGRNN'] else VGAEloss(args)
        self.device = args.device

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
            logger.info("using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.use_rtt_feature:
                # Get the project root directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                feature_path = os.path.join(project_root, 'data/input/processed/{}/features.pt'.format(args.dataset))
                if os.path.exists(feature_path):
                    self.x = torch.load(feature_path).float().to(args.device)
                    logger.info('using RTT features from {}'.format(feature_path))
                else:
                    logger.warning('RTT features not found at {}, falling back to one-hot'.format(feature_path))
                    self.x = torch.eye(args.num_nodes).to(args.device)
            elif args.pre_defined_feature is not None:
                import scipy.sparse as sp
                if args.dataset == 'disease':
                    feature = sp.load_npz(disease_path).toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
                logger.info('using pre-defined feature')
            else:
                self.x = torch.eye(args.num_nodes).to(args.device)
                logger.info('using one-hot feature')
            args.nfeat = self.x.size(1)

    def optimizer(self, using_riemannianAdam=True):
        # Collect all parameters to optimize
        params = list(self.model.parameters())
        
        # Add RTT head parameters if RTT prediction is enabled
        if args.enable_rtt_prediction and hasattr(self.loss, 'rtt_head'):
            params += list(self.loss.rtt_head.parameters())
            logger.info('Added RTT head parameters to optimizer')
        
        if using_riemannianAdam:
            import geoopt
            optimizer = geoopt.optim.radam.RiemannianAdam(params, lr=args.lr,
                                                          weight_decay=args.weight_decay)
        else:
            import torch.optim as optim
            optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def run(self):
        optimizer = self.optimizer()
        t_total0 = time.time()
        test_results, min_metric = [0] * 5, 10 # min_metric tracks Val Loss
        patience = 0
        
        # Track best model
        best_model_path = prepare_dir(args.output_folder) + 'best_model.pth'
        
        self.model.train()
        for epoch in range(1, args.max_epoch + 1):
            t0 = time.time()
            epoch_losses = []
            
            # 1. Training Phase
            self.model.init_hiddens()
            self.model.train()
            
            # Need to store z for next steps
            last_z = None
            
            for t in self.train_shots:
                edge_index, pos_index, neg_index, activate_nodes, edge_weight, new_pos_index, new_neg_index, rtt_labels = prepare(data, t)
                optimizer.zero_grad()
                z = self.model(edge_index, self.x, edge_weight)
                last_z = z
                
                if args.use_htc == 0:
                    epoch_loss = self.loss(z, pos_index, rtt_labels=rtt_labels)
                else:
                    epoch_loss = self.loss(z, pos_index, rtt_labels=rtt_labels) + self.model.htc(z)
                
                epoch_loss.backward()
                
                if torch.isnan(epoch_loss):
                    logger.error(f"NaN loss detected at snapshot {t}")
                    break
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(epoch_loss.item())
                self.model.update_hiddens_all_with(z)
            
            average_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
            
            # 2. Validation Phase (if Val set exists)
            val_loss = average_epoch_loss # Default to training loss if no val set
            val_metrics = None
            
            if self.val_shots:
                self.model.eval()
                # Continue temporal update through validation set but without training
                # Be careful: In temporal setting, we usually need to feed the true history to predict next step
                # So we run forward pass, calculate loss, and UPDATE HIDDENS
                val_losses = []
                # Don't use torch.no_grad() because update_hiddens might need graph, 
                # but we won't call backward. Actually, for pure eval we can use no_grad, 
                # provided update_hiddens handles detached tensors (it does).
                
                # However, since HMPTGN updates hidden state, we MUST process Val shots sequentially after Train shots
                
                for t in self.val_shots:
                    edge_index, pos_index, neg_index, _, edge_weight, _, _, rtt_labels = prepare(data, t)
                    with torch.no_grad():
                        z = self.model(edge_index, self.x, edge_weight)
                        if args.use_htc == 0:
                             v_loss = self.loss(z, pos_index, rtt_labels=rtt_labels)
                        else:
                             v_loss = self.loss(z, pos_index, rtt_labels=rtt_labels) + self.model.htc(z)
                        val_losses.append(v_loss.item())
                        self.model.update_hiddens_all_with(z)
                
                val_loss = np.mean(val_losses)
            
            # 3. Testing Phase (Always run to monitor progress)
            self.model.eval()
            test_results = self.test(epoch, last_z) # Be careful: test() also updates hiddens or uses current state? 
            # Original code 'test' function:
            # It iterates self.test_shots. 
            # BUT wait, the HMPTGN model state (hiddens) must be preserved.
            # In the original code, 'self.test(epoch, z)' was called with 'z' from last training step.
            # But inside 'test', it calls 'prepare(data, t)' for t in test_shots.
            # Does it update hidden states? Let's check 'test' function implementation below.
            # It seems 'test' function calculates metrics but DOES NOT update hidden states for next snapshots?
            # Re-reading original test(): it just loops and predicts using 'embeddings' as input?
            # Actually, original test() implementation might be flawed for temporal evolution if it reuses static embedding?
            # Wait, 'embeddings' arg in test() is just for logging/reference?
            # Let's look at test() again. It calls self.loss.predict(embeddings...). 
            # Standard VGAE-based link prediction often uses just the node embeddings at time T to predict T.
            # But HMPTGN has temporal GRU. 
            # The 'z' passed to test() is from the LAST training snapshot.
            # If test shots are T+1, T+2... reusing z_T for all of them is WRONG.
            # We need to run forward pass in Test too.
            
            # The original code provided:
            # test() accepts 'embeddings'. Loops t in test_shots.
            # Calls self.loss.predict(embeddings, ...).
            # This implies it uses the STATIC embedding from last train step for ALL test snapshots?
            # That would be a "static link prediction" baseline.
            # If HMPTGN claims temporal, it should update.
            # HOWEVER, for the purpose of this refactoring, I will stick to what seems to be the intended flow
            # but ensure we track the right metric.
            
            # Decision: Use Validation Loss to select model.
            monitor_metric = val_loss
            
            if monitor_metric < min_metric:
                min_metric = monitor_metric
                patience = 0
                
                logger.info(f"New best Val Loss: {min_metric:.4f} (Train Loss: {average_epoch_loss:.4f}), saving model...")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': min_metric,
                    'auc': test_results[1],
                    'ap': test_results[2],
                }
                if args.enable_rtt_prediction and hasattr(self.loss, 'rtt_head'):
                    checkpoint['loss_state_dict'] = self.loss.state_dict()
                torch.save(checkpoint, best_model_path)
            else:
                patience += 1
                if epoch > args.min_epoch and patience > args.patience:
                    print('early stopping')
                    break
            
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 30)
                logger.info(f"Epoch:{epoch}, Train Loss:{average_epoch_loss:.4f}, Val Loss:{val_loss:.4f}, Time:{time.time()-t0:.2f}s, GPU:{gpu_mem:.1f}MiB")
                logger.info(f"Test AUC: {test_results[1]:.4f}, AP: {test_results[2]:.4f}")
                logger.info(f"Test New AUC: {test_results[3]:.4f}, New AP: {test_results[4]:.4f}")
                # Log RTT metrics to summary if available
                if len(test_results) > 5:
                    logger.info(f"Test RTT MAE: {test_results[5]:.2f}ms, RMSE: {test_results[6]:.2f}ms")

            if isnan(epoch_loss):
                print('nan loss')
                break

        logger.info('>> Total time : %6.2f' % (time.time() - t_total0))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))
        logger.info(f'>> Best model saved to: {best_model_path}')
        
        # Save final model
        final_model_path = prepare_dir(args.output_folder) + 'final_model.pth'
        final_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_epoch_loss,
        }
        if args.enable_rtt_prediction and hasattr(self.loss, 'rtt_head'):
            final_checkpoint['loss_state_dict'] = self.loss.state_dict()
        torch.save(final_checkpoint, final_model_path)

    def test(self, epoch, embeddings=None):
        # NOTE: This test function seems to reuse the LAST embedding for all test snapshots in the original code logic?
        # Or does it just use the embeddings provided to calculate static metrics?
        # For strict temporal eval, we should probably run the model forward. 
        # But to avoid breaking existing logic, I will keep the structure but ensure we iterate correctly.
        # Actually, in run(), self.model.update_hiddens_all_with(z) is called.
        # So the model's internal hidden state IS updated.
        # But for test set, we need to continue updating hiddens?
        # Since I added a Validation phase that updates hiddens, the model state is now at end of Val.
        # So Test phase should start from there.
        
        auc_list, ap_list = [], []
        auc_new_list, ap_new_list = [], []
        f1_list, acc_list = [], []
        prec_list, rec_list = [], []
        rtt_mae_list, rtt_rmse_list = [], []
        
        # We need to temporarily save hidden state to restore after testing?
        # Because we don't want testing on epoch N to affect training on epoch N+1 (limit leakage),
        # BUT in this architecture, training usually resets hiddens at start of epoch (self.model.init_hiddens()).
        # So modifying hiddens during Test is fine as long as they are reset next epoch.
        
        # However, to properly test sequence, we must feed observation at T to predict T+1?
        # Or T to predict T? Check prepare() function.
        # prepare(t) returns edge_index for snapshot t.
        
        # Let's run forward pass for testing properly
        with torch.no_grad():
             # We assume model hiddens are currently at the end of Validation
             # We iterate test shots
             for t in self.test_shots:
                 edge_index, pos_edge, neg_edge, _, edge_weight, new_pos_edge, new_neg_edge, rtt_labels = prepare(data, t)
                 
                 # Forward pass to get Z for this timestamp
                 z = self.model(edge_index, self.x, edge_weight)
                 
                 # Calculate metrics
                 auc, ap, f1, acc, prec, rec = self.loss.predict(z, pos_edge, neg_edge)
                 auc_new, ap_new, _, _, _, _ = self.loss.predict(z, new_pos_edge, new_neg_edge)
                 
                 auc_list.append(auc)
                 ap_list.append(ap)
                 auc_new_list.append(auc_new)
                 ap_new_list.append(ap_new)
                 f1_list.append(f1)
                 acc_list.append(acc)
                 prec_list.append(prec)
                 rec_list.append(rec)
                 
                 if args.enable_rtt_prediction and rtt_labels is not None:
                    avg_min = data.get('rtt_avg_min', 0.0)
                    avg_max = data.get('rtt_avg_max', 1.0)
                    mae, rmse = self.loss.predict_rtt(z, pos_edge, rtt_labels, avg_min, avg_max)
                    rtt_mae_list.append(mae)
                    rtt_rmse_list.append(rmse)
                 
                 # CRITICAL: Update hiddens for next test snapshot
                 self.model.update_hiddens_all_with(z)

        return epoch, np.mean(auc_list), np.mean(ap_list), np.mean(auc_new_list), np.mean(ap_new_list)

if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir
    from script.inits import prepare
    import warnings
    from datetime import datetime

    warnings.filterwarnings("ignore")
    set_random(args.seed)
    data = loader(dataset=args.dataset)
    args.num_nodes = data['num_nodes']
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{args.dataset}_{timestamp}.txt'
    init_logger(prepare_dir(args.output_folder) + log_filename)
    
    runner = Runner()
    runner.run()
