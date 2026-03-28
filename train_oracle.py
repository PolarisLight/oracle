"""
This is the main file for training CIFAR dataset.
This file is used to test this framework.
"""
from argparse import Namespace, ArgumentParser
from utils.utils import set_seed
import warnings
warnings.filterwarnings("ignore")
# =============================================================================
# need to be redefined for every new task
from config.config_cifar_base import config as base_config
from config.config_cifar_oracle import config as config
import models.Resnet as models
import datasets.OBC306 as datasets
# =============================================================================

def get_model_and_dataset(args:Namespace)->tuple:
    """
    Get the model and dataset.
    This function need to be redefined for every new task.
    """
    
    dataset = {'train': getattr(datasets, args.dataset.name)(args, train=True),
               'val':getattr(datasets, args.dataset.name)(args, train=False)}

    args.label_dis = dataset['train'].get_cls_num_list()
    args.tail_classes = dataset['train'].get_tail_classes(50)
    model = getattr(models, args.model.name)(args)
    return model, dataset

def main(args:Namespace)->None:
    """
    Main function
    args: arguments
    """
    model, dataset = get_model_and_dataset(args)
    
    from Trainer.moe_trainer import MoE_Trainer
    trainer = MoE_Trainer(args, model, dataset, args.train.device)
    trainer.train()
    

# =============================================================================
# Normal Mode
# =============================================================================

def Normal_Mode():
    opt = ArgumentParser()
    opt.add_argument('--task', type=str, default='oracle-20k', help='task name')
    opt.add_argument('--model', type=str, default='ResNet_MoE', help='model name')
    opt.add_argument('--dataset', type=str, default='OBC306', help='dataset name')
    opt.add_argument('--seed', type=int, default=123, help='random seed')
    opt.add_argument('--save_log', type=bool, default=0, help='save log')
    opt = opt.parse_args()
    set_seed(opt.seed)
    args = config(opt.task, opt.model, opt.dataset, opt.save_log)
    args.base_args = base_config(opt.task, opt.model, opt.dataset, opt.save_log)
    # args.train.use_wandb = False  # Disable Weights & Biases logging
    main(args)


import nni
from loguru import logger
def Abl_Mode():
    opt = ArgumentParser()
    opt.add_argument('--task', type=str, default='oracle-20k', help='task name')
    opt.add_argument('--model', type=str, default='ResNet_MoE', help='model name')
    opt.add_argument('--dataset', type=str, default='OBC306', help='dataset name')
    opt.add_argument('--seed', type=int, default=123, help='random seed')
    opt.add_argument('--save_log', type=bool, default=0, help='save log')
    opt = opt.parse_args()

    args = config(opt.task, opt.model, opt.dataset, opt.save_log)
    args.base_args = base_config(opt.task, opt.model, opt.dataset, opt.save_log)
    args.train.use_wandb = False  # Disable Weights & Biases logging
    args.train.use_nni = True  # Enable NNI for hyperparameter tuning
    args.core_params.mamba_alpha = 0.3
    if hasattr(args.train,'use_nni') and args.train.use_nni:
        try:
            tuner_params = nni.get_next_parameter()
            if 'lr' in tuner_params:
                args.train.lr = tuner_params['lr']
            if 'lr_rt' in tuner_params:
                args.train.lr_rt = tuner_params['lr_rt']
            if 'batch_size' in tuner_params:
                args.train.batch_size = tuner_params['batch_size']
            if 'weight_decay' in tuner_params:
                args.train.weight_decay = tuner_params['weight_decay']
            if 'optimizer' in tuner_params:
                args.train.optimizer = tuner_params['optimizer']
            if 'scheduler' in tuner_params:
                args.scheduler.name = tuner_params['scheduler']
            if 'decay_rate' in tuner_params:
                args.scheduler.decay_rate = tuner_params['decay_rate']
            if 'decay_epoch' in tuner_params:
                args.scheduler.decay_epoch = tuner_params['decay_epoch']
            if 'seed' in tuner_params:
                set_seed(tuner_params['seed'])
            else:
                set_seed(opt.seed)
            if 'alpha' in tuner_params:
                args.core_params.mamba_alpha = tuner_params['alpha']
            if 'conf_start_epoch' in tuner_params:
                args.core_params.conf_start_epoch = tuner_params['conf_start_epoch']
            if 'num_experts' in tuner_params:
                args.model.num_experts = tuner_params['num_experts']
        except Exception as e:
            logger.error(f"Error in getting tuner parameters: {e}")
            raise e
    main(args)

use_abl_mode = 1

if __name__ == '__main__':
    if use_abl_mode:
        Abl_Mode()
    else:
        Normal_Mode()