import os
import csv
import time
import numpy as np
import scipy.io
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.autograd import grad
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import ProgressBarBase

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.factory import get_reference_directions, get_visualization, get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

use_mlflow = "USE_MLFLOW" in os.environ and os.environ["USE_MLFLOW"] == 'true'
if use_mlflow:
    from pytorch_lightning.loggers import MLFlowLogger
else:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
    #import logging
    #logger = logging.getLogger("wandb")
    #logger.setLevel(logging.ERROR) # avoid excess logging
    wandb.login();

class TextProgressBar(ProgressBarBase):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if trainer.current_epoch == 0:
            self.start_time = time.time()
        if trainer.current_epoch % 10 == 0:
            percent = (trainer.current_epoch / trainer.max_epochs) * 100.0
            speed = 0.0 if trainer.current_epoch == 0 else trainer.current_epoch * 60 / (time.time() - self.start_time)
            print("%3.2f%% epoch=%d/%d speed=%.0f/min" % (percent, trainer.current_epoch, trainer.max_epochs, speed), end="\r")

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=64):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        if batch_size == -1 or self.dataset_len < batch_size:
            batch_size = self.dataset_len
        self.batch_size = batch_size

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class BurgersDataset(TensorDataset):
    def __init__(self, path='../data/Burgers/', normalize=True):
        data = scipy.io.loadmat(path + 'burgers_shock.mat')
        x = np.tile(data['x'], (data['t'].shape[0],1)) # TN x 1
        t = np.repeat(data['t'], data['x'].shape[0], axis=0) # TN x 1
        X = np.concatenate([x,t], axis=1) # TN x 2

        Y = data['usol'].T.reshape(-1,1) # TN x 1

        if normalize:
            self.Y_mean = np.mean(Y, axis=0)
            Y -= self.Y_mean
            self.Y_std = np.std(Y, axis=0)
            Y /= self.Y_std

        super().__init__(torch.from_numpy(X).float(), torch.from_numpy(Y).float())

class WavesDataset(TensorDataset):
    def __init__(self, path='../data/Waves_Square/', normalize=True):
        X = np.loadtxt(path + "X_star.txt") # 2 x N
        T = np.loadtxt(path + "T_star.txt") # T
        U = np.loadtxt(path + "U_star.txt") # N x T  
        V = np.loadtxt(path + "H_0.txt")    # N

        xx = np.tile(X.T, (T.shape[0],1)) # TN x 2
        tt = np.repeat(T.reshape(-1,1), X.shape[1], axis=0) # TN x 1
        X = np.concatenate([xx,tt], axis=1) # TN x 3

        uu = U.T.reshape(-1,1) # TN x 1
        vv = np.tile(V.reshape(-1,1), (T.shape[0],1)) # TN x 1
        Y = np.concatenate([uu,vv], axis=1) # TN x 2

        if normalize:
            self.Y_mean = np.mean(Y, axis=0)
            Y -= self.Y_mean
            self.Y_std = np.std(Y, axis=0)
            Y /= self.Y_std

        super().__init__(torch.from_numpy(X).float(), torch.from_numpy(Y).float())

class AdvDifDataset(TensorDataset):
    def __init__(self, path='../data/AdvDif/', normalize=True):
        x_data = np.loadtxt(path + "X_star.txt") # N x 2  
        t_data = np.loadtxt(path + "T_star.txt") # T
        T_data = np.loadtxt(path + "Temp.txt")   # N x T

        xy = np.tile(x_data, (t_data.shape[0],1)) # TN x 2
        t = np.repeat(t_data.reshape(-1,1), x_data.shape[0], axis=0) # TN x 1
        X = np.concatenate([xy,t], axis=1) # TN x 3

        Y = T_data.T.reshape(-1,1) # TN x 1

        if normalize:
            self.Y_mean = np.mean(Y, axis=0)
            Y -= self.Y_mean
            self.Y_std = np.std(Y, axis=0)
            Y /= self.Y_std

        super().__init__(torch.from_numpy(X).float(), torch.from_numpy(Y).float())

class SWEDataset(TensorDataset):
    def __init__(self, path='../data/SWE/', normalize=True):
        x_data = np.loadtxt(path + "X_star.txt")   # 2 x N  
        t_data = np.loadtxt(path + "T_star.txt")   # T
        u_data = np.loadtxt(path + "U0_SWE.txt")   # N x T
        v_data = np.loadtxt(path + "U1_SWE.txt")   # N x T
        h_data = np.loadtxt(path + "eta.txt")      # N x T

        # downsample
        #x_data = x_data[:,::8]
        #t_data = t_data[::10]
        #u_data = u_data[::8,::10]
        #v_data = v_data[::8,::10]
        #h_data = h_data[::8,::10]

        xy = np.tile(x_data.T, (t_data.shape[0],1)) # TN x 2
        t = np.repeat(t_data.reshape(-1,1), x_data.shape[1], axis=0) # TN x 1
        X = np.concatenate([xy,t], axis=1) # TN x 3

        u = u_data.T.reshape(-1,1) # TN x 1
        v = v_data.T.reshape(-1,1) # TN x 1
        h = h_data.T.reshape(-1,1) # TN x 1
        Y = np.concatenate([u,v,h], axis=1) # TN x 3

        if normalize:
            self.Y_mean = np.mean(Y, axis=0)
            Y -= self.Y_mean
            self.Y_std = np.std(Y, axis=0)
            Y /= self.Y_std

        super().__init__(torch.from_numpy(X).float(), torch.from_numpy(Y).float())

class DataModule(LightningDataModule):
    def __init__(self, dataset, val_split=0.1, test_split=0.0, batch_size=-1):
        super().__init__()
        self.dataset = dataset
        self.n_val = int(val_split*len(self.dataset))
        self.n_test = int(test_split*len(self.dataset))
        self.n_train = len(self.dataset) - self.n_val - self.n_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train, self.val, self.test = random_split(self.dataset, [self.n_train, self.n_val, self.n_test])

    def train_dataloader(self):
        dataset = self.train.dataset
        indices = self.train.indices
        return FastTensorDataLoader(*[t[indices] for t in dataset.tensors], batch_size=self.batch_size)

    def val_dataloader(self):
        dataset = self.val.dataset
        indices = self.val.indices
        return FastTensorDataLoader(*[t[indices] for t in dataset.tensors], batch_size=self.batch_size)

    def test_dataloader(self):
        dataset = self.test.dataset
        indices = self.test.indices
        return FastTensorDataLoader(*[t[indices] for t in dataset.tensors], batch_size=self.batch_size)

class BurgersPDE:
    def __init__(self, lambda1=1.0, lambda2=0.01/np.pi):
        self.name = 'burgers'
        self.input_width = 2
        self.output_width = 1
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.Y_mean = None
        self.Y_std = None

    def __str__(self):
        return self.name
        
    def forward(self, x, t, u):
        if self.Y_mean is not None and self.Y_std is not None:
            u = u*self.Y_std[0] + self.Y_mean[0]

        u_t  = grad(u,   t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_x  = grad(u,   x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = grad(u_x, x, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        return u_t + self.lambda1*u*u_x - self.lambda2*u_xx
    
class WavesPDE:
    def __init__(self): 
        self.name = 'waves'
        self.input_width = 3
        self.output_width = 2

        self.Y_mean = None
        self.Y_std = None

    def __str__(self):
        return self.name
    
    def forward(self, x, y, t, u, v):
        if self.Y_mean is not None and self.Y_std is not None:
            u = u*self.Y_std[0] + self.Y_mean[0]
            v = v*self.Y_std[0] + self.Y_mean[0]

        u_x  = grad(u,   x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = grad(u_x, x, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        u_y  = grad(u,   y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_yy = grad(u_y, y, create_graph=True, grad_outputs=torch.ones_like(u_y))[0]
        u_t  = grad(u,   t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_tt = grad(u_t, t, create_graph=True, grad_outputs=torch.ones_like(u_t))[0]
        v_x  = grad(v,   x, create_graph=True, grad_outputs=torch.ones_like(v))[0]
        v_y  = grad(v,   y, create_graph=True, grad_outputs=torch.ones_like(v))[0]        
        return u_tt - v*(u_xx + u_yy) - (v_x*u_x + v_y*u_y)
    
class AdvDifPDE:
    def __init__(self, D=0.02, u=np.cos(22.5*np.pi/180.0), v=np.sin(22.5*np.pi/180.0)):
        self.name = 'advdif'
        self.input_width = 3
        self.output_width = 1
        self.D = D
        self.u = u
        self.v = v

        self.Y_mean = None
        self.Y_std = None

    def __str__(self):
        return self.name
    
    def forward(self, x, y, t, T):
        if self.Y_mean is not None and self.Y_std is not None:
            T = T*self.Y_std[0] + self.Y_mean[0]
        
        T_x  = grad(T,   x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        T_xx = grad(T_x, x, create_graph=True, grad_outputs=torch.ones_like(T_x))[0]
        T_y  = grad(T,   y, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        T_yy = grad(T_y, y, create_graph=True, grad_outputs=torch.ones_like(T_y))[0]
        T_t  = grad(T,   t, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return T_t - self.D*(T_xx + T_yy) + self.u*T_x + self.v*T_y

class SWEPDE:
    def __init__(self, H=10.0, g=9.81):
        self.name = 'swe'
        self.input_width = 3
        self.output_width = 3
        self.H = H
        self.g = g

        self.Y_mean = None
        self.Y_std = None

    def __str__(self):
        return self.name

    def forward(self, x, y, t, u, v, h):
        if self.Y_mean is not None and self.Y_std is not None:
            u = u*self.Y_std[0] + self.Y_mean[0]
            v = v*self.Y_std[1] + self.Y_mean[1]
            h = h*self.Y_std[2] + self.Y_mean[2]
        
        h_x  = grad(h, x, create_graph=True, grad_outputs=torch.ones_like(h))[0]
        h_y  = grad(h, y, create_graph=True, grad_outputs=torch.ones_like(h))[0]
        h_t  = grad(h, t, create_graph=True, grad_outputs=torch.ones_like(h))[0]
        u_x  = grad(u, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_t  = grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        v_y  = grad(v, y, create_graph=True, grad_outputs=torch.ones_like(v))[0]
        v_t  = grad(v, t, create_graph=True, grad_outputs=torch.ones_like(v))[0]
        f1 = h_t + self.H * (u_x + v_y)
        f2 = u_t + self.g * h_x
        f3 = v_t + self.g * h_y
        
        return F.mse_loss(f1, torch.zeros_like(f1)) +\
               F.mse_loss(f2, torch.zeros_like(f2)) +\
               F.mse_loss(f3, torch.zeros_like(f3))

class PINN(LightningModule):
    def __init__(self, pde, layer_widths=[100,100], activation_function='tanh',
                 weight_initializer='none', learning_rate=1e-3, alpha=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        self.pde = pde
        self.input_width = pde.input_width
        self.output_width = pde.output_width
        self.learning_rate = learning_rate
        self.alpha = alpha
        
        sizes = [self.input_width] + layer_widths + [self.output_width]
        self.net = nn.Sequential(
            *[self.block(dim_in, dim_out, activation_function)
            for dim_in, dim_out in zip(sizes[:-1], sizes[1:-1])],
            nn.Linear(sizes[-2], sizes[-1]) # output layer is regular linear transformation
        )
        
        if weight_initializer == 'xavier':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
            self.net.apply(init_weights)
                
    def forward(self, x):
        return self.net.forward(x)
    
    def block(self, dim_in, dim_out, activation_function):
        activation_functions = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['relu', nn.ReLU()],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            ['softplus', nn.Softplus()],
            ['softsign', nn.Softsign()],
            ['tanhshrink', nn.Tanhshrink()],
            ['celu', nn.CELU()],
            ['gelu', nn.GELU()],
            ['elu', nn.ELU()],
            ['selu', nn.SELU()],
            ['logsigmoid', nn.LogSigmoid()]
        ])
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            activation_functions[activation_function],
        )
        
    def loss(self, Xu, Yu, Xf=None):
        data_loss = F.mse_loss(self.forward(Xu), Yu)
        
        if Xf is not None:
            Xf.requires_grad=True
            X = [Xf[:,i] for i in range(self.input_width)]
            Xf = torch.stack(X,1)
            
            Y_hat = self.forward(Xf)
            Y = [Y_hat[:,i] for i in range(self.output_width)]
            
            f = self.pde.forward(*X, *Y)
            phys_loss = F.mse_loss(f, torch.zeros_like(f))            
            return data_loss, phys_loss
        return data_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def training_step(self, batch, batch_idx):
        X, Y = batch
        data, phys = self.loss(X, Y, X)
        loss = (1.0-self.alpha)*data + self.alpha*phys
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        data, phys = self.loss(X, Y, X)
        loss = (1.0-self.alpha)*data + self.alpha*phys
        self.log("val_data_loss", data)
        self.log("val_phys_loss", phys)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        X, Y = batch
        data, phys = self.loss(X, Y, X)
        loss = (1.0-self.alpha)*data + self.alpha*phys
        self.log("test_data_loss", data)
        self.log("test_phys_loss", phys)
        self.log("test_loss", loss)
        return loss

class ProblemAlpha(ElementwiseProblem):
    def __init__(self, mopinn):
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=[0.0], xu=[1.0])
        self.mopinn = mopinn

    def _evaluate(self, x, out, *args, **kwargs):
        algorithm = kwargs['algorithm']

        alpha = x[0]
        res, _ = self.mopinn.train_net(alpha, generation=algorithm.n_gen)
        out['F'] = res

        np.save('checkpoint-' + self.mopinn.pde.name + '.npy', algorithm)

class ProblemLayerWidths(ElementwiseProblem):
    def __init__(self, mopinn):
        super().__init__(n_var=4, n_obj=3, n_constr=0, xl=[0.0, 1, 0, 0], xu=[1.0, 10, 10, 10])
        self.mopinn = mopinn

    def _evaluate(self, x, out, *args, **kwargs):
        algorithm = kwargs['algorithm']

        alpha = x[0]
        layer_widths = [x[1]*10]
        if x[2] != 0:
            layer_widths.append(min(x[2]*10, x[1]*10))
            if x[3] != 0:
                layer_widths.append(min(x[3]*10, x[2]*10, x[1]*10))
        res, _ = self.mopinn.train_net(alpha, generation=algorithm.n_gen, layer_widths=layer_widths)

        # penalize number of neurons
        if algorithm.n_gen < 5:
            res.append(1.0)
        else:
            res.append(np.prod(layer_widths))
        out['F'] = res

        np.save('checkpoint-' + self.mopinn.pde.name + '.npy', algorithm)

class ProblemActivationFunction(ElementwiseProblem):
    def __init__(self, mopinn):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=[0.0, 0], xu=[1.0, 11])
        self.mopinn = mopinn
        self.activation_functions = ['lrelu', 'relu', 'tanh', 'sigmoid', 'softplus', 'softsign',
            'tanhshrink', 'celu', 'gelu', 'elu', 'selu', 'logsigmoid']

    def _evaluate(self, x, out, *args, **kwargs):
        algorithm = kwargs['algorithm']

        alpha = x[0]
        activation_function = self.activation_functions[x[1]]
        res, _ = self.mopinn.train_net(alpha, generation=algorithm.n_gen,
                                       activation_function=activation_function)
        out['F'] = res

        np.save('checkpoint-' + self.mopinn.pde.name + '.npy', algorithm)

class ProblemArchitecture(ElementwiseProblem):
    def __init__(self, mopinn):
        super().__init__(n_var=5, n_obj=3, n_constr=0, xl=[0.0, 1, 0, 0, 0], xu=[1.0, 10, 10, 10, 11])
        self.mopinn = mopinn
        self.activation_functions = ['lrelu', 'relu', 'tanh', 'sigmoid', 'softplus', 'softsign',
            'tanhshrink', 'celu', 'gelu', 'elu', 'selu', 'logsigmoid']

    def _evaluate(self, x, out, *args, **kwargs):
        algorithm = kwargs['algorithm']

        alpha = x[0]
        layer_widths = [x[1]*10]
        if x[2] != 0:
            layer_widths.append(min(x[2]*10, x[1]*10))
            if x[3] != 0:
                layer_widths.append(min(x[3]*10, x[2]*10, x[1]*10))
        activation_function = self.activation_functions[x[4]]
        res, _ = self.mopinn.train_net(alpha, generation=algorithm.n_gen, layer_widths=layer_widths,
                                       activation_function=activation_function)

        # penalize number of neurons
        if algorithm.n_gen < 5:
            res.append(1.0)
        else:
            res.append(np.prod(layer_widths))
        out['F'] = res

        np.save('checkpoint-' + self.mopinn.pde.name + '.npy', algorithm)

class MOPINN:
    def __init__(self, pde, dataset, layer_widths=[100, 100], activation_function='tanh',
                 weight_initializer='none', learning_rate=1e-3, val_split=0.1, test_split=0.0,
                 batch_size=-1, deterministic=True): 
        self.pde = pde
        self.dataset = dataset
        self.layer_widths = layer_widths
        self.activation_function = activation_function
        self.weight_initializer = weight_initializer
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.deterministic = deterministic

        self.unique_name = 'default'
        self.method = None
        self.problem = None
        self.generations = None
        self.trainer_kwargs = {}

        if hasattr(self.dataset, 'Y_mean') and hasattr(self.dataset, 'Y_std'):
            self.pde.Y_mean = self.dataset.Y_mean
            self.pde.Y_std = self.dataset.Y_std

        print("Dataset size: %d" % len(dataset))

    def train_net(self, alpha, generation=0, log=True, layer_widths=None, activation_function=None, weight_initializer=None, learning_rate=None, unique_name=None, **kwargs):
        min_delta = 0.0
        if 'min_delta' in self.trainer_kwargs:
            min_delta = self.trainer_kwargs['min_delta']
        if 'min_delta' in kwargs:
            min_delta = kwargs['min_delta']
        trainer_kwargs = {
            'callbacks': [
                TextProgressBar(),
                EarlyStopping(monitor='val_loss', patience=1000, mode='min', min_delta=min_delta),
            ],
            'min_epochs': 1000,
            'log_every_n_steps': 1,
            'enable_checkpointing': False,
            'enable_model_summary': False,
            'enable_progress_bar': True,
        }
        for key, val in self.trainer_kwargs.items():
            if key not in ['min_delta']:
                trainer_kwargs[key] = val
        for key, val in kwargs.items():
            if key not in ['min_delta']:
                trainer_kwargs[key] = val

        if layer_widths is None:
            layer_widths = self.layer_widths
        if activation_function is None:
            activation_function = self.activation_function
        if weight_initializer is None:
            weight_initializer = self.weight_initializer
        if learning_rate is None:
            learning_rate = self.learning_rate
        if unique_name is None:
            unique_name = self.unique_name

        if self.deterministic:
            torch.manual_seed(7981)
            rng = np.random.default_rng(7981)

        datamodule = DataModule(self.dataset,
            val_split=self.val_split, 
            test_split=self.test_split, 
            batch_size=self.batch_size,
        )

        model = PINN(
            pde=self.pde, 
            layer_widths=layer_widths,
            activation_function=activation_function,
            weight_initializer=weight_initializer,
            learning_rate=learning_rate,
            alpha=alpha,
        )

        if log:
            logger = WandbLogger(name='alpha_%.4f'%(alpha,))
            logger.log_hyperparams({
                'method': self.method,
                'problem': self.problem,
                'generation': generation,
                'unique_name': unique_name,
            })

        trainer = Trainer(logger=logger, **trainer_kwargs)
        trainer.tune(model, datamodule=datamodule)

        print("Start: gen=%d alpha=%f architecture=%s×%s" % (generation, alpha, layer_widths, activation_function))
        start_time = time.time()
        trainer.fit(model, datamodule=datamodule)
        duration = time.time() - start_time

        if trainer.interrupted:
            raise KeyboardInterrupt

        if log:
            logger.experiment.finish()

        res = trainer.logged_metrics
        val_data_loss = res['val_data_loss'] if 'val_data_loss' in res else 0.0
        val_phys_loss = res['val_phys_loss'] if 'val_phys_loss' in res else 0.0
        epoch = trainer.current_epoch
        print("End: epoch=%d loss_data=%f loss_physics=%f dur=%ds" % (epoch, val_data_loss, val_phys_loss, duration))

        return [val_data_loss, val_phys_loss], model

    def train(self, problem='alpha', generations=10, population=25, method='MOEAD', load_checkpoint=False, unique_name='default', **kwargs):
        print('Problem:', self.pde.name)

        self.trainer_kwargs = kwargs
        self.unique_name = unique_name
        self.method = method
        self.problem = problem
        self.generations = generations
        if problem == 'architecture':
            print('Training: alpha, layer_widths, and activation_function')
            mask = ["real", "int", "int", "int", "int"]
            problem = ProblemArchitecture(self)
        elif problem == 'activation_function':
            print('Training: alpha and activation_function')
            mask = ["real", "int"]
            problem = ProblemActivationFunction(self)
        elif problem == 'layer_widths':
            print('Training: alpha and layer_widths')
            mask = ["real", "int", "int", "int"]
            problem = ProblemLayerWidths(self)
        else:
            self.problem = 'alpha'
            print('Training: alpha')
            mask = ["real"]
            problem = ProblemAlpha(self)

        if load_checkpoint and os.path.isfile('checkpoint-' + self.pde.name + '.npy'):
            algorithm, = np.load('checkpoint-' + self.pde.name + '.npy', allow_pickle=True).flatten()
        else:
            sampling = MixedVariableSampling(mask, {
                "real": get_sampling("real_random"),
                "int": get_sampling("int_random")
            })

            crossover = MixedVariableCrossover(mask, {
                "real": get_crossover("real_sbx", prob=0.9, eta=3.0),
                "int": get_crossover("int_sbx", prob=0.9, eta=3.0)
            })

            mutation = MixedVariableMutation(mask, {
                "real": get_mutation("real_pm", eta=20.0),
                "int": get_mutation("int_pm", eta=20.0)
            })

            if method == 'MOEAD':
                ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
                algorithm = MOEAD(
                    ref_dirs,
                    n_neighbors=population,
                    prob_neighbor_mating=0.7,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                )
            elif method == 'NSGA2':
                algorithm = NSGA2(
                    pop_size=population,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                )
            elif method == 'NSGA3':
                ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
                algorithm = NSGA3(
                    ref_dirs,
                    pop_size=population,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                )
            else:
                raise ValueError('unknown method ' + method)

            print('Method:', method)

        try:
            res = minimize(
                problem,
                algorithm,
                ('n_gen', generations),
                seed=1,
                verbose=True,
            )
        except KeyboardInterrupt:
            print('ERROR: keyboard interrupt')

        wandb.finish()
