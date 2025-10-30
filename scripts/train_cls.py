import argparse
import torch
import json
import os

from neurobert.src.model.model_utils import load_clean_state_dict_to_model
from neurobert.src.train_test.base import Trainer, Tester
from neurobert.src.utils import validate_and_prepare_start_config, get_device
from neurobert.src.config import ModelConfig, TrainerConfig
from neurobert.src.data_utils.cls_utils import get_mice_sessions, create_dataset


def main(args):
    print('Prepare configs')
    config = validate_and_prepare_start_config(args.config, args.mode, script_mode='cls')

    path_to_WT = config['path_to_WT']
    path_to_5xFAD = config['path_to_5xFAD']
    path_to_model = config['path_to_model']
    path_to_model_cfg = config['path_to_model_cfg']
    path_to_trainer_cfg = config['path_to_trainer_cfg']
    train_indexes = config.get('train_indexes')
    val_indexes = config.get('val_indexes')
    test_indexes = config.get('test_indexes')    
    save_path = config['save_path']

    # load model, trainer and mask config
    model_cfg = ModelConfig.from_json(path_to_model_cfg)
    trainer_cfg = TrainerConfig.from_json(path_to_trainer_cfg)

    # create datasets and dataloaders
    print('Create Dataloaders')
    n_workers = args.num_workers
    start_mode = args.mode

    mice_wt = get_mice_sessions(path_to_WT, class_label=0)
    mice_5xFAD = get_mice_sessions(path_to_5xFAD, class_label=1)
    labeled_mice = mice_wt | mice_5xFAD

    if start_mode == 'train':
        if train_indexes != None:
            train_dict = {}
            for idx in train_indexes:
                train_dict[idx] = labeled_mice[idx]
            train_dataset = create_dataset(train_dict, array_format=args.array_format)
            print(f'--- Train mice: {train_indexes}')
        else:
            train_dataset = create_dataset(labeled_mice, array_format=args.array_format)
            print(f'--- Train mice: {list(labeled_mice.keys())}')

        data_loader_train = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=trainer_cfg.batch_size, 
                                                        shuffle=True, 
                                                        num_workers=n_workers,  
                                                        persistent_workers=True)
        
        if val_indexes != None:
            val_dict = {}
            for idx in val_indexes:
                val_dict[idx] = labeled_mice[idx]
            val_dataset = create_dataset(val_dict, array_format=args.array_format)
            data_loader_val = torch.utils.data.DataLoader(val_dataset, 
                                                        batch_size=trainer_cfg.batch_size, 
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
            test_dataset = create_dataset(test_dict, array_format=args.array_format)
            print(f'--- Test mice: {test_indexes}')
        else:
            test_dataset = create_dataset(labeled_mice, array_format=args.array_format)
            print(f'--- Test mice: {list(labeled_mice.keys())}')

        data_loader_test = torch.utils.data.DataLoader(test_dataset, 
                                                       batch_size=trainer_cfg.batch_size, 
                                                       shuffle=False, 
                                                       num_workers=n_workers,  
                                                       persistent_workers=True)
    else:
        raise ValueError('Wrong start mode. Only "train" / "test"')

    print('Dataloaders was created\n')

    # model loading
    if start_mode == 'train':
        model = load_clean_state_dict_to_model(model_type='classifier_for_fine_tune',
                                            cfg=model_cfg,
                                            checkpoint_path=path_to_model)
        print('Pretrained model was loaded\n')
    elif start_mode == 'test':
        model = load_clean_state_dict_to_model(model_type='classifier_for_inference',
                                               cfg=model_cfg, 
                                               checkpoint_path=path_to_model)
        print('Fine-tuned for testing model was loaded\n')

    device = get_device(None)
    session_probs= args.session_probs
    if start_mode == 'train':
        optimizer = torch.optim.Adam(params=[{'params': model.bert.embed.parameters(), 'lr': trainer_cfg.lr*0.8**3}, # /2
                                            {'params': model.bert.blocks[0].parameters(), 'lr': trainer_cfg.lr*0.9**3}, # /2
                                            {'params': model.bert.blocks[1].parameters(), 'lr': trainer_cfg.lr*0.9**2}, # /2
                                            {'params': model.bert.blocks[2].parameters(), 'lr': trainer_cfg.lr*0.9**1}, 
                                            {'params': model.bert.blocks[3].parameters(), 'lr': trainer_cfg.lr}, 
                                            {'params': model.classification_blocks.parameters(), 'lr': trainer_cfg.lr_cl}], 
                                    weight_decay=trainer_cfg.weight_decay)
        if trainer_cfg.freez_mode == 'embedded':
            print('FREEZE EMBED PARAMS')
            for param in model.bert.embed.parameters():
                param.requires_grad = False
        elif trainer_cfg.freez_mode == 'full':
            print('FREEZE EMBED BERT PARAMS')
            for param in model.bert.parameters():
                param.requires_grad = False
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trainer_cfg.sheduler_gamma)

        with open(os.path.join(save_path, 'trainer_cfg.json'), 'w') as f:
            json.dump(trainer_cfg._asdict(), f)
        with open(os.path.join(save_path, 'model.json'), 'w') as f:
            json.dump(model_cfg._asdict(), f)
        
        fine_tune = Trainer(model, 
                    optimizer, 
                    device, 
                    trainer_cfg.n_epochs,
                    save_steps=trainer_cfg.save_steps, 
                    scheduler=scheduler,
                    save_path=save_path)
        
        fine_tune.train(mode='binary_classification', 
                                          data_loader_train=data_loader_train, 
                                          data_loader_val=data_loader_val,
                                          data_parallel=True, 
                                          class_labels=[0, 1], 
                                          class_weights_train=None,
                                          session_probs=session_probs)
        print('Model was fine-tuned!')

    if test_indexes != None:
        tester = Tester(model, 
                        data_loader_test,
                        device, 
                        save_path=save_path)
        
        tester.test(class_labels=[0, 1], 
                    session_probs=session_probs,
                    mode='binary_classification')
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
