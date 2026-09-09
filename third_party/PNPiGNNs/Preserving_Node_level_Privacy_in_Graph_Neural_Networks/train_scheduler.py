                    
import enum
import torch
import time
from tqdm import tqdm
import random
from functorch import make_functional_with_buffers
from functorch import vmap, grad
from copy import deepcopy
import os
import numpy as np
import math
import logging
import json
from pathlib import Path

''' '''
import utils

from privacy import accounting_analysis as aa
from privacy import sampling as sampling
import datasets.SETUP as SETUP
''' helpers '''
TRAIN_SETUP_LIST = ('epoch', 'device', 'optimizer', 'loss_metric', 'enable_per_grad')

class Phase(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()
    PHASE_to_PHASE_str = { TRAIN: "Training", VAL: "Validation", TEST: "Testing"}

class trainer:
    def __init__(self, *, model, optimizer, loaders, device, criterion, args,
                 target_delta, degree_bound, steps):
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.device = device
        self.criterion = criterion
        self.args = args
        self.steps = int(steps)
        self.scheduled_train_rounds = 0

        self.worker_model_func, self.worker_param_func, self.worker_buffers_func = make_functional_with_buffers(deepcopy(model), disable_autograd_tracking=True)
        self.record_data_type = 'weighted_recall'
        q = args.q = self.args.expected_batchsize / len(self.train_loader.dataset)
        self.args.delta = target_delta
        self.std, self.achieved_epsilon = aa.get_std_node_dp(
            q=q,
            steps=self.steps,
            D_out=degree_bound,
            M_train=self.args.num_neighbors,
            epsilon=self.args.priv_epsilon,
            delta=self.args.delta,
            saving_path=Path(self.args.cache_dir) / 'privacy_node_dp',
            device=self.device,
        )
        self.args.std = self.std
        self.dataset_size = len(self.train_loader.dataset)
        # exit() 

        self.dataset_size = len(self.train_loader.dataset)

        ''' prepare for logging '''
        self.is_info_initialized = False
        self.init_log(dir = args.log_dir)
        para_info, self.args.num_params = utils.show_param(model)
        self.write_log( para_info, verbose=False) 
        arg_info = self._arg_readable(self.args)
        self.write_log(f'{arg_info}')
        self.write_log( f'dataset: {self.train_loader.dataset.graph_data_name}' )

        ''' '''
        self.data_logger = utils.data_recorder(root = self.args.log_dir)
        self.data_logger.record_data(f'weighted_recall.csv', arg_info)
        self.data_logger.record_data(f'weighted_recall.csv', para_info)
        self.data_logger.record_data(f'weighted_recall.csv', self.train_loader.dataset.graph_data_name)
        self.data_logger.record_data(f'weighted_recall.csv', self.train_loader.dataset.graph_data)

        self.write_log(self.train_loader.dataset.graph_data_name)
        self.write_log(self.train_loader.dataset.graph_data)

        ''' json data recorder template '''
        self.json_recorder = json_data_recorder(f'jd_{self.train_loader.dataset.graph_data_name}_{self.args.graph_setting}_eps{args.priv_epsilon}.json')
        self.json_recorder.add_record('args', vars(self.args))
        self.json_recorder.add_record('num_params', self.args.num_params)
        self.json_recorder.add_record('dataset', self.train_loader.dataset.graph_data_name)
        self.json_recorder.add_record('train_acc', None)
        self.json_recorder.add_record('val_acc', None)
        self.json_recorder.add_record('test_acc', None)
        self.json_recorder.add_record('test_pre', None)

    def _arg_readable(self, arg):
        arg_dict = vars(self.args)
        arg_info = ['args:'] + [f'{key} -- {value}' for key, value in arg_dict.items()]
        arg_info = f'\n{"="*40}\n' + '\n'.join(arg_info) + f'\n{"="*40}\n'
        return arg_info

    def init_log(self, dir = 'logs'):
        Path('/'.join(__file__.split('/')[:-1]) + f'/{dir}').mkdir(parents=True, exist_ok=True)
        self.is_info_initialized = True
        file_name = '/'.join(__file__.split('/')[:-1]) + f'/{dir}/log.txt'
        # print('==> log file name is at: ', file_name)
        logging.basicConfig(
            filename = file_name, 
            filemode = 'a',
            datefmt = "%H:%M:%S", 
            level = logging.INFO,
            format = '%(asctime)s[%(levelname)s] ~ %(message)s'
        )
        self.write_log('\n\n' + "VV" * 40 + '\n' + " " * 37 + 'NEW LOG\n' + "^^" * 40)  
    
    def write_log(self, info = None, c_tag = None, verbose = True):
        if not self.is_info_initialized:
            self.init_log()
            self.is_info_initialized = True
        if verbose:
            print(str(info))
        if c_tag is not None:
            logging.info(str(c_tag) + ' ' + str(info))
        else:
            logging.info(str(info))

    def run(self):
        start_time = time.time()
        best_validation_accuracy = -float("inf")
        best_state = None
        best_validation_metrics = None
        for epoch in range(self.args.epoch):
            self.write_log(f'\nEpoch: [{epoch}] '.ljust(11) + '#' * 35)
            self.epoch = epoch
            train_metrics = self.one_epoch(train_or_val=Phase.TRAIN, loader=self.train_loader)
            self.json_recorder.add_record('train_acc', float(train_metrics.__getattr__(self.record_data_type)))
            validation_metrics = self.one_epoch(train_or_val=Phase.VAL, loader=self.val_loader)
            validation_accuracy = float(validation_metrics.hit_accuracy)
            self.json_recorder.add_record('val_acc', validation_accuracy)
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_validation_metrics = validation_metrics
                best_state = deepcopy(self.model.state_dict())
            self.data_logger.record_data(
                f'{self.record_data_type}.csv',
                f'{epoch}   {float(train_metrics.__getattr__(self.record_data_type)) * 100:.2f}%'
                f'   {float(validation_metrics.__getattr__(self.record_data_type)) * 100:.2f}%',
            )
        if best_state is not None:
            self.model.load_state_dict(best_state)
            for p_model, p_worker in zip(self.model.parameters(), self.worker_param_func):
                p_worker.copy_(p_model.data)
        test_metrics = self.one_epoch(train_or_val=Phase.TEST, loader=self.test_loader)
        self.json_recorder.add_record('test_acc', float(test_metrics.hit_accuracy))
        self.json_recorder.add_record('test_pre', float(test_metrics.__getattr__('weighted_precis')))
        self.write_log(f'\n\n=> TIME for ALL: {time.time() - start_time:.2f}  secs')
        self._clear_loader()
        self.json_recorder.save()
        return best_validation_metrics, test_metrics

    def _clear_loader(self):
        for loader in (self.train_loader, self.val_loader, self.test_loader):
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()

    def one_epoch(self, train_or_val, loader):
        metrics = utils.ClassificationMetrics(num_classes = self.args.num_classes)
        metrics.num_images = metrics.loss = 0 
        is_training = train_or_val is Phase.TRAIN
        self.model.train(is_training)

        ''' per example method '''
        def compute_loss(model_para, buffers, x, targets):
            # print(f'==> x shape: {x.shape}, targets shape: {targets.shape}')
            predictions = self.worker_model_func(model_para, buffers, x)

            ''' only use the first node's prediction '''
            predictions, targets = predictions[:1], targets[:1]
            ''' use the center node's prediction '''
            # predictions, targets = predictions[::self.args.num_neighbors+1], targets[::self.args.num_neighbors+1]

            # print(f'==> predictions shape: {predictions.shape}')
            loss = self.criterion(predictions, targets.flatten()) #* inputs.shape[0]
            return loss
        def per_forward(model_para, buffers, x):
            predictions = self.worker_model_func(model_para, buffers, x)
            ''' only use the first node's prediction '''
            # predictions = predictions[:1]
            return predictions

        def manual_forward(x, targets):
            # print(f'==> x shape: {x.shape}, edge_index shape: {edge_index.shape}, targets shape: {targets.shape}')
            out = vmap(per_forward, in_dims=(None, None, 0) )(self.worker_param_func, self.worker_buffers_func, x)
            # print(f'==> out shape: {out.shape}, targets shape: {targets.shape}')
            out = out[:,0,:]
            targets = targets[:,0]
            # print(f'==> out shape: {out.shape}, targets shape: {targets.shape}')
            loss = self.criterion(out, targets.view(-1))
            return loss, out, targets

        s = time.time()
        if is_training: 
            print(f'==> have {len(loader)} iterations in this epoch')
            s_time = time.time()
            for i, batch in enumerate(loader):
                self.scheduled_train_rounds += 1
                if batch is None:
                    continue
                x, targets = batch
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                per_grad = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(
                    self.worker_param_func, self.worker_buffers_func, x, targets,
                )
                loss, out, targets = manual_forward(x, targets)
                metrics.batch_update(loss, out, targets)
                # print(f'==> calculation time: {time.time()-c_time:.5f} secs')
                # c_time = time.time()
                self.other_routine(per_grad)
                # print(f'==> calculation time: {time.time()-c_time:.5f} secs')
                # s_time = time.time()

        else:
            for i, batch in enumerate(loader):
                if batch is None:
                    continue
                x, targets = batch
                x, targets = x.to(self.device), targets.to(self.device)
                loss, out, targets = manual_forward(x, targets)
                metrics.batch_update(loss, out, targets)

        metrics.loss /= max(metrics.num_images, 1)
        self.write_log(f'    {train_or_val}: {time.time()-s:.3f} S, {self.record_data_type} = {float(metrics.__getattr__(self.record_data_type))*100:.2f}%')

        return metrics

    def other_routine(self, per_grad):

        per_grad = self.clip_per_grad(per_grad)


        ''' forming gradients'''
        for p_model, p_per in zip(self.model.parameters(), per_grad):
            p_model.grad = torch.mean(p_per, dim=0)
            ''' add noise to gradients'''
            p_model.grad = p_model.grad + torch.randn_like(p_model.grad) * self.std * self.args.C / p_per.shape[0] 
            

            ''' clamp extreme gradients'''
            cth = self.std * self.args.C / p_per.shape[0] / 1e2
            p_model.grad = torch.clamp(p_model.grad, -cth, cth)

        self.model_update()
    
    def model_update(self):
        '''update parameters'''
        self.optimizer.step()  
        ''' copy parameters to worker '''
        for p_model, p_worker in zip(self.model.parameters(), self.worker_param_func):
            p_worker.copy_(p_model.data)
    
    def clip_per_grad(self, per_grad):
        per_grad = list(per_grad)
        per_grad_norm = 2 * self._compute_per_grad_norm(per_grad) + 1e-6 
        # print(f'==> per_grad_norm shape: {per_grad_norm.shape}')
        # print(f'==> per_grad_norm: {per_grad_norm}')

        ''' clipping/normalizing '''
        multiplier = torch.clamp(self.args.C / per_grad_norm, max = 1)
        for index, p in enumerate(per_grad):
            pass
            ''' normalizing '''
            # per_grad[index] = p * self._make_broadcastable(self.args.C / per_grad_norm, p) 
            ''' clipping '''
            per_grad[index] = p * self._make_broadcastable( multiplier, p ) 
        return per_grad

    def _compute_per_grad_norm(self, iterator, which_norm = 2):
        all_grad = torch.cat([p.reshape(p.shape[0], -1) for p in iterator], dim = 1)
        per_grad_norm = torch.norm(all_grad, dim = 1, p = which_norm)

        return per_grad_norm
    
    def _make_broadcastable(self, tensor_to_be_reshape, target_tensor):
        broadcasting_shape = (-1, *[1 for _ in target_tensor.shape[1:]])
        return tensor_to_be_reshape.reshape(broadcasting_shape)
    
class json_data_recorder:
    def __init__(self, filename):
        self.data_dict = {}
        self.dir_name = 'data_records'
        
        self._set_record_filename(filename)
        
    def _set_record_filename(self, filename='data_record.json'):
        self.record_name = filename
        dir_path = '/'.join(__file__.split('/')[:-1]) + f'/{self.dir_name}'
        os.mkdir(dir_path) if not os.path.exists(dir_path) else None
        self.file_path =  dir_path + f'/{filename}'

    def add_record(self, name, data):
        if name not in self.data_dict:
            self.data_dict[name] = []
        if data is not None:
            self.data_dict[name].append(data)

    def save(self):
        data = []
        
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        
        with open(self.file_path, 'w') as f:
            json.dump(data + [self.data_dict], f, indent=4) 

def get_data_from_record(filename):
    path =  '/'.join(__file__.split('/')[:-1]) + f'/data_records/{filename}'
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    dr = json_data_recorder()
    dr.set_record_filename('sampling_noise.json')
    dr.add_record('a', 1)
    dr.add_record('a', 2)
    dr.save()
    data = get_data_from_record('sampling_noise.json')
    print(data)