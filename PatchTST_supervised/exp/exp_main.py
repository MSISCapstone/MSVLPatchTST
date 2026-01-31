from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            # Use direct checkpoint_path if provided, otherwise construct from setting
            if hasattr(self.args, 'checkpoint_path') and self.args.checkpoint_path:
                ckpt_path = self.args.checkpoint_path
            else:
                ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            print(f'Checkpoint path: {ckpt_path}')
            self.model.load_state_dict(torch.load(ckpt_path))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save - use output folder structure
        git_root = os.popen('git rev-parse --show-toplevel 2>/dev/null').read().strip()
        if not git_root:
            git_root = os.getcwd()
        
        # Extract model_id from setting (e.g., "weather_336_96_PatchTST..." -> "weather_336_96")
        parts = setting.split('_')
        if len(parts) >= 3:
            model_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
        else:
            model_id = setting
        
        folder_path = os.path.join(git_root, 'output', 'Original', 'test_results', model_id) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        print(f'Results saved to: {folder_path}')
        
        # Save results file
        results_file = folder_path + 'results.txt'
        with open(results_file, 'w') as f:
            f.write(f'Original PatchTST Test Results\n')
            f.write('='*80 + '\n')
            f.write(f'Setting: {setting}\n')
            f.write(f'MSE: {mse:.6f}\n')
            f.write(f'MAE: {mae:.6f}\n')
            f.write(f'RMSE: {rmse:.6f}\n')
            f.write(f'RSE: {rse:.6f}\n')
            f.write('='*80 + '\n')

        # Save predictions and ground truth
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        print(f'Saved predictions shape: {preds.shape}')
        print(f'Saved ground truth shape: {trues.shape}')
        return

    def test_sliding_window(self, setting, test=0, num_iterations=4):
        """
        Sliding window multi-step prediction test.
        
        Each iteration uses REAL historical data (not previous predictions) to predict 96 steps.
        The prediction windows are consecutive, covering num_iterations * pred_len total steps.
        
        Example with num_iterations=4, seq_len=336, pred_len=96:
            Step 1: Input [0:336]    → Predict [336:432]   (ground truth available)
            Step 2: Input [96:432]   → Predict [432:528]   (ground truth available)
            Step 3: Input [192:528]  → Predict [528:624]   (ground truth available)
            Step 4: Input [288:624]  → Predict [624:720]   (ground truth available)
        """
        test_data, test_loader = self._get_data(flag='test')
        pred_len = self.args.pred_len
        seq_len = self.args.seq_len
        total_pred_len = num_iterations * pred_len
        
        print(f'\nSliding window prediction: {num_iterations} iterations x {pred_len} steps = {total_pred_len} total steps')
        print(f'Each prediction uses real {seq_len}-step lookback window')
        
        if test:
            print('loading model')
            if hasattr(self.args, 'checkpoint_path') and self.args.checkpoint_path:
                ckpt_path = self.args.checkpoint_path
            else:
                ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            print(f'Checkpoint path: {ckpt_path}')
            self.model.load_state_dict(torch.load(ckpt_path))

        # Access raw data from dataset
        data_x = test_data.data_x  # Full normalized data [total_len, channels]
        data_y = test_data.data_y  # Full normalized data for targets
        
        total_len = len(data_x)
        
        # Calculate how many complete sliding window sequences we can create
        required_len = seq_len + num_iterations * pred_len
        num_samples = total_len - required_len + 1
        
        if num_samples <= 0:
            raise ValueError(f"Dataset too short. Need {required_len} timesteps, have {total_len}")
        
        print(f"Total data length: {total_len}, Creating {num_samples} sliding window samples")
        
        f_dim = -1 if self.args.features == 'MS' else 0
        n_features = data_y.shape[1] if f_dim == 0 else 1
        
        all_preds = []
        all_trues = []

        self.model.eval()
        with torch.no_grad():
            for sample_idx in range(num_samples):
                iteration_preds = []
                iteration_trues = []
                
                for iteration in range(num_iterations):
                    # Calculate the start position for this iteration's input window
                    input_start = sample_idx + iteration * pred_len
                    input_end = input_start + seq_len
                    
                    # Calculate the target position (what we're predicting)
                    target_start = input_end
                    target_end = target_start + pred_len
                    
                    if target_end > total_len:
                        break
                    
                    # Get input sequence
                    seq_x = data_x[input_start:input_end]  # [seq_len, channels]
                    seq_x = torch.FloatTensor(seq_x).unsqueeze(0).to(self.device)  # [1, seq_len, channels]
                    
                    # Get ground truth
                    if f_dim == 0:
                        true_y = data_y[target_start:target_end, :]  # [pred_len, channels]
                    else:
                        true_y = data_y[target_start:target_end, -1:]  # [pred_len, 1] for MS
                    
                    # Forward pass
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(seq_x)
                    else:
                        # For encoder-decoder models, need marks
                        # This simplified version assumes PatchTST
                        outputs = self.model(seq_x)
                    
                    pred = outputs[:, -pred_len:, f_dim:].cpu().numpy()[0]  # [pred_len, features]
                    
                    iteration_preds.append(pred)
                    iteration_trues.append(true_y)
                
                if len(iteration_preds) == num_iterations:
                    full_pred = np.concatenate(iteration_preds, axis=0)  # [total_pred_len, features]
                    full_true = np.concatenate(iteration_trues, axis=0)  # [total_pred_len, features]
                    
                    all_preds.append(full_pred)
                    all_trues.append(full_true)
                
                if sample_idx % 1000 == 0:
                    print(f"  Processed {sample_idx}/{num_samples} samples...")
        
        all_preds = np.array(all_preds)  # [N, total_pred_len, features]
        all_trues = np.array(all_trues)  # [N, total_pred_len, features]
        
        print(f'\nSliding Window Evaluation complete:')
        print(f'  Predictions shape: {all_preds.shape}')
        print(f'  Ground truth shape: {all_trues.shape}')
        
        # Calculate metrics on full predictions (all 384 steps have ground truth!)
        mae, mse, rmse, mape, mspe, rse, corr = metric(all_preds, all_trues)
        print(f'\nTest Metrics (all {total_pred_len} steps):')
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        
        # Per-iteration metrics
        print(f'\nPer-iteration Metrics:')
        for i in range(num_iterations):
            start_idx = i * pred_len
            end_idx = (i + 1) * pred_len
            iter_preds = all_preds[:, start_idx:end_idx, :]
            iter_trues = all_trues[:, start_idx:end_idx, :]
            iter_mae, iter_mse, _, _, _, _, _ = metric(iter_preds, iter_trues)
            print(f'  Iteration {i+1} (steps {start_idx}-{end_idx}): MSE={iter_mse:.7f}, MAE={iter_mae:.7f}')
        
        # Save results
        git_root = os.popen('git rev-parse --show-toplevel 2>/dev/null').read().strip()
        if not git_root:
            git_root = os.getcwd()
        
        parts = setting.split('_')
        if len(parts) >= 3:
            model_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
        else:
            model_id = setting
        
        folder_path = os.path.join(git_root, 'output', 'Original', 'test_results', model_id) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        print(f'Results saved to: {folder_path}')
        
        # Save results file
        results_file = folder_path + 'results.txt'
        with open(results_file, 'w') as f:
            f.write(f'Original PatchTST Test Results (Sliding Window)\n')
            f.write('='*80 + '\n')
            f.write(f'Setting: {setting}\n')
            f.write(f'Sliding Window: {num_iterations} x {pred_len} = {total_pred_len} total steps\n')
            f.write(f'Each prediction uses real {seq_len}-step lookback\n')
            f.write(f'MSE: {mse:.6f}\n')
            f.write(f'MAE: {mae:.6f}\n')
            f.write(f'RMSE: {rmse:.6f}\n')
            f.write(f'RSE: {rse:.6f}\n')
            f.write('='*80 + '\n')

        # Save predictions and ground truth (both are 384 steps now!)
        np.save(folder_path + 'pred.npy', all_preds)
        np.save(folder_path + 'true.npy', all_trues)
        print(f'Saved predictions shape: {all_preds.shape}')
        print(f'Saved ground truth shape: {all_trues.shape}')
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
