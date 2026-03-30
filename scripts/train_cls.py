import argparse
import torch
import json
import os

from neurobert.src.model.model_utils import load_clean_state_dict_to_model
from neurobert.src.train_test.base import Trainer, Tester
from neurobert.src.utils import validate_and_prepare_start_config, get_device, set_seed
from neurobert.src.config import AnyConfig
from neurobert.src.data_utils.cls_utils import get_mice_sessions, create_dataset
from neurobert.src.data_utils.datasets import Preprocess4Mask, AugDataset4FineTune

def main(args):
    print('Prepare configs')
    config = validate_and_prepare_start_config(args.config, args.mode, script_mode='cls')

    path_to_WT = config['path_to_WT']
    path_to_5xFAD = config['path_to_5xFAD']
    path_to_model = config['path_to_model']
    path_to_cfg = config['path_to_cfg']
    train_indexes = config.get('train_indexes')
    val_indexes = config.get('val_indexes')
    test_indexes = config.get('test_indexes')    
    save_path = config['save_path']

    # load model, trainer and mask config
    cfg = AnyConfig.from_yaml(path=path_to_cfg)

    # create datasets and dataloaders
    print('Create Dataloaders')
    n_workers = args.num_workers
    start_mode = args.mode
    set_seed(cfg.dataset.seed)

    mice_wt = get_mice_sessions(path_to_WT, class_label=0)
    mice_5xFAD = get_mice_sessions(path_to_5xFAD, class_label=1)
    labeled_mice = mice_wt | mice_5xFAD

    if cfg.train.task == 'multitask':
        mask_creator = Preprocess4Mask(cfg.mask)
        multitask_weigths = torch.Tensor([cfg.train.cls_weight, cfg.train.recon_mask_weight, cfg.train.recon_full_weight])
    else:
        multitask_weigths = None

    if start_mode == 'train':
        if train_indexes != None:
            train_dict = {}
            for idx in train_indexes:
                train_dict[idx] = labeled_mice[idx]
            train_dataset = create_dataset(train_dict, array_format=cfg.dataset.array_format)
            print(f'--- Train mice: {train_indexes}')
        else:
            train_dataset = create_dataset(labeled_mice, array_format=cfg.dataset.array_format)
            print(f'--- Train mice: {list(labeled_mice.keys())}')

        if cfg.train.task == 'multitask':
            weights = train_dataset.count_weigths(method=cfg.dataset.weight_method,
                                      beta=cfg.dataset.beta)
            train_dataset = AugDataset4FineTune(train_dataset, [mask_creator])

        data_loader_train = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=cfg.train.batch_size, 
                                                        shuffle=True, 
                                                        num_workers=n_workers,  
                                                        persistent_workers=True)
        
        if val_indexes != None:
            val_dict = {}
            for idx in val_indexes:
                val_dict[idx] = labeled_mice[idx]
            val_dataset = create_dataset(val_dict, array_format=cfg.dataset.array_format)

            if cfg.train.task == 'multitask':
                val_dataset = AugDataset4FineTune(val_dataset, [mask_creator])

            data_loader_val = torch.utils.data.DataLoader(val_dataset, 
                                                        batch_size=cfg.train.batch_size, 
                                                        shuffle=False, 
                                                        num_workers=n_workers,  
                                                        persistent_workers=True)
            print(f'--- Val mice: {val_indexes}')
        else:
            data_loader_val = None
        
    elif start_mode == 'test' or test_indexes != None:
        if test_indexes != None:
            test_dict = {}
            for idx in test_indexes:
                test_dict[idx] = labeled_mice[idx]
            test_dataset = create_dataset(test_dict, array_format=cfg.dataset.array_format)
            print(f'--- Test mice: {test_indexes}')
        else:
            test_dataset = create_dataset(labeled_mice, array_format=cfg.dataset.array_format)
            print(f'--- Test mice: {list(labeled_mice.keys())}')

        if cfg.train.task == 'multitask':
            test_dataset = AugDataset4FineTune(test_dataset, [mask_creator])

        data_loader_test = torch.utils.data.DataLoader(test_dataset, 
                                                       batch_size=cfg.train.batch_size, 
                                                       shuffle=False, 
                                                       num_workers=n_workers,  
                                                       persistent_workers=True)
    else:
        raise ValueError('Wrong start mode. Only "train" / "test"')

    print('Dataloaders was created\n')

    # model loading
    if start_mode == 'train':
        model = load_clean_state_dict_to_model(model_type='classifier_for_fine_tune',
                                            cfg=cfg.model,
                                            checkpoint_path=path_to_model)
        model._initialize(only_head=True)
        print('Pretrained model was loaded\n')
    elif start_mode == 'test':  
        model = load_clean_state_dict_to_model(model_type='classifier_for_inference',
                                               cfg=cfg.model, 
                                               checkpoint_path=path_to_model)
        print('Fine-tuned for testing model was loaded\n')

    device = get_device(None)
    session_probs= args.session_probs
    if start_mode == 'train':
        optimizer = torch.optim.Adam(params=[{'params': model.bert.embed.parameters(), 'lr': cfg.train.lr}, # /2
                                            {'params': model.bert.blocks[0].parameters(), 'lr': cfg.train.lr}, # /2
                                            {'params': model.bert.blocks[1].parameters(), 'lr': cfg.train.lr}, # /2
                                            {'params': model.bert.blocks[2].parameters(), 'lr': cfg.train.lr}, 
                                            {'params': model.bert.blocks[3].parameters(), 'lr': cfg.train.lr}, 
                                            {'params': model.classification_blocks.parameters(), 'lr': cfg.train.lr_cl}], 
                                    weight_decay=cfg.optim.weight_decay)
        if cfg.train.freez_mode == 'embedded':
            print('FREEZE EMBED PARAMS')
            for param in model.bert.embed.parameters():
                param.requires_grad = False
        elif cfg.train.freez_mode == 'full':
            print('FREEZE EMBED BERT PARAMS')
            for param in model.bert.parameters():
                param.requires_grad = False
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler.sheduler_gamma)

        # with open(os.path.join(save_path, 'cfg.train.json'), 'w') as f:
        #     json.dump(cfg.train._asdict(), f)
        # with open(os.path.join(save_path, 'model.json'), 'w') as f:
        #     json.dump(cfg.model._asdict(), f)

        cfg.to_yaml(os.path.join(save_path, 'config.yml'))
        
        fine_tune = Trainer(model, 
                    optimizer, 
                    device, 
                    cfg.train.n_epochs,
                    save_steps=cfg.train.save_steps, 
                    scheduler=scheduler,
                    save_path=save_path)
        
        fine_tune.train(mode=cfg.train.task, 
                        data_loader_train=data_loader_train, 
                        data_loader_val=data_loader_val,
                        data_parallel=True, 
                        class_labels=[0, 1], 
                        class_weights_train=weights,
                        multitask_weigths=multitask_weigths,
                        session_probs=session_probs)
        print('Model was fine-tuned!')

    if test_indexes != None:
        tester = Tester(model, 
                        data_loader_test,
                        device, 
                        save_path=save_path)
        
        tester.test(class_labels=[0, 1], 
                    session_probs=session_probs,
                    mode=cfg.train.task)
        print('Model was tested!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to main config JSON')
    parser.add_argument('-a', '--array_format', type=str, default='mem', help='Format to load data ("mem" - memmap/"arr" - arrays)')
    parser.add_argument('-m', '--mode', type=str, default='train', help='train/test')
    parser.add_argument('-n', '--num_workers', type=int, default=2, help='Number of workers in dataloader')
    parser.add_argument('-s', '--session_probs', type=int, choices=[0, 1], default=1, help='If 1 (True), counts probabilities on each session')
    args = parser.parse_args()
    args.session_probs = bool(args.session_probs)

    print('INFO:')
    max_len = max(len(arg) for arg in vars(args))
    for arg in vars(args):
        print(f"{arg:<{max_len}} : {getattr(args, arg)}")
    print()

    main(args)
