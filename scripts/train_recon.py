import argparse
import torch
import json
import os

from neurobert.src.model.model_utils import load_clean_state_dict_to_model, init_weights
from neurobert.src.train_test.base import Trainer, Tester
from neurobert.src.utils import load_data, validate_and_prepare_start_config, split_dataset, get_device
from neurobert.src.config import MaskConfig, ModelConfig, TrainerConfig
from neurobert.src.data_utils.datasets import Preprocess4Mask, Dataset4Pretrain


def main(args):
    config = validate_and_prepare_start_config(args.config, args.mode, script_mode='recon')

    path_to_data = config['path_to_data']
    path_to_model = config['path_to_model']
    path_to_model_cfg = config['path_to_model_cfg']
    path_to_trainer_cfg = config['path_to_trainer_cfg']
    path_to_mask_cfg = config['path_to_mask_cfg']
    save_path = config['save_path']

    # load model, trainer and mask config
    model_cfg = ModelConfig.from_json(path_to_model_cfg)
    trainer_cfg = TrainerConfig.from_json(path_to_trainer_cfg)
    if path_to_mask_cfg:
        mask_cfg = MaskConfig.from_json(path_to_mask_cfg)
    else:
        mask_cfg = MaskConfig.from_dict({'mask_ratio': 0.15, 
                                         'mask_alpha': 6, 
                                         'max_gram': 10, 
                                         'mask_prob': 0.8})

    # create datasets and dataloaders
    print('Create Dataloaders')
    n_workers = args.num_workers
    dataset, _ = load_data(path_to_data, array_format=args.array_format)
    data, shapes = dataset
    
    start_mode = args.mode
    p = args.part_to_split
    mask_creator = Preprocess4Mask(mask_cfg)
    if start_mode == 'train':
        if len(shapes) == 1:
            print('Can split only to train/val, only one file was given, but for testing need to use another session!')
            train_val_dataset = Dataset4Pretrain(data, shapes, [mask_creator])
            train_dataset, val_dataset = split_dataset(train_val_dataset, p, seed=trainer_cfg.seed)
            test_dataset = None
        else:
            train_val_data, test_data = split_dataset(data, p, seed=trainer_cfg.seed)
            train_val_shapes, test_shapes = split_dataset(shapes, p, seed=trainer_cfg.seed)
            train_val_dataset = Dataset4Pretrain(train_val_data, train_val_shapes, [mask_creator])
            test_dataset = Dataset4Pretrain(test_data, test_shapes, [mask_creator])
            train_dataset, val_dataset = split_dataset(train_val_dataset, p, seed=trainer_cfg.seed)

        data_loader_train = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size=trainer_cfg.batch_size, 
                                                        shuffle=True, 
                                                        num_workers=n_workers,  
                                                        persistent_workers=True)
        data_loader_val = torch.utils.data.DataLoader(val_dataset, 
                                                      batch_size = trainer_cfg.batch_size, 
                                                      shuffle=True,
                                                      num_workers=n_workers,  
                                                      persistent_workers=True)
        
    elif start_mode == 'test':
        mask_creator = Preprocess4Mask(mask_cfg)
        test_dataset = Dataset4Pretrain(data, shapes, [mask_creator])
    else:
        raise ValueError('Wrong mode. Expected: "tran" or "test"')

    data_loader_test = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=trainer_cfg.batch_size, 
                                                   shuffle=False,
                                                   num_workers=n_workers,
                                                   persistent_workers=True)
    print('Dataloaders was created\n')

    # model loading
    if start_mode == 'train':
        model = load_clean_state_dict_to_model(model_type='bert',
                                            cfg=model_cfg,
                                            checkpoint_path=None)
        model.apply(init_weights)
        print('Empty model was initialized')
    elif start_mode == 'test':
        model = load_clean_state_dict_to_model(model_type='bert',
                                               cfg=model_cfg, 
                                               checkpoint_path=path_to_model)
        print('Pretrained for testing model was loaded')

    # Trining
    device = get_device(None)
    if start_mode == 'train':
        with open(os.path.join(save_path, 'trainer_cfg.json'), 'w') as f:
            json.dump(trainer_cfg._asdict(), f)
        with open(os.path.join(save_path, 'model_cfg.json'), 'w') as f:
            json.dump(model_cfg._asdict(), f)
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=trainer_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trainer_cfg.sheduler_gamma)
        
        pretrain = Trainer(model, 
                        optimizer, 
                        device, 
                        trainer_cfg.n_epochs,
                        save_steps=trainer_cfg.save_steps, 
                        scheduler=scheduler,
                        save_path=save_path)
        
        pretrain.train(mode='pretrain_reconstruction', 
                       data_loader_train=data_loader_train, 
                       data_loader_val=data_loader_val,
                       data_parallel=True)
        
        print('Model was pretrained!')

    if test_dataset:
        tester = Tester(model, data_loader_test, device, save_path=save_path)
        tester.test(mode='pretrain_reconstruction')
        print('Model was tested!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to main config JSON')
    parser.add_argument('-a', '--array_format', type=str, default='mem', help='Format to load data ("mem" - memmap/"arr" - arrays)')
    parser.add_argument('-m', '--mode', type=str, default='train', help='train/test')
    parser.add_argument('-n', '--num_workers', type=int, default=2, help='Number of workers in dataloader')
    parser.add_argument('-p', '--part_to_split', type=float, default=0.9, help='Data split part (ex: train = 0.9 / val = 0.1) <=> -p 0.9')
    args = parser.parse_args()
    
    print('INFO:')
    max_len = max(len(arg) for arg in vars(args))
    for arg in vars(args):
        print(f"{arg:<{max_len}} : {getattr(args, arg)}")
    print()

    main(args)
