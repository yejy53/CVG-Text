import os
import yaml
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")

from dataset import OSMTextImageDataset
from utils import setup_logging
from model_loader import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP with Distributed Data Parallel")
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')

    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--expand', action='store_true')

    parser.add_argument('--img_type', type=str, choices=['sat', 'osm', 'stv'], required=True, 
                        help='Choose the data type: sat, osm, stv')
    parser.add_argument('--logging', action='store_true', help='Enable logging and model saving')
    # DDP
    parser.add_argument('--init_method', default='env://',help="init-method")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #-----------------------------------------------------------------------------#
    # Config                                                                      #
    #-----------------------------------------------------------------------------#
    
    args = parse_args()

    version, model_name, expand, img_type, log = args.version, args.model, args.expand, args.img_type, args.logging

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config['paths'].items():
            if isinstance(value, str) and '{version}' in value:
                value = value.format(version=version)
            globals()[key] = value

    safe_model_name = model_name.replace('/', '_')
    hyperparams = f"{version}_lr_{args.lr}_bs_{args.batch_size}_expand_{args.expand}_{args.img_type}"
    if log:
        setup_logging(log_dir, safe_model_name, hyperparams)
        
    #-----------------------------------------------------------------------------#
    # DDP                                                                         #
    #-----------------------------------------------------------------------------#
    
    dist.init_process_group(backend='nccl', init_method=args.init_method)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK"))
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", local_rank)
    

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    
    model, preprocessor, evaluater, forward = load_model(model_name,
                                                checkpoint_path='',
                                                expand_text=expand,
                                                is_stv = img_type == 'stv'
                                                )
    
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    model = DDP(model, device_ids=[rank])
        
    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
    if img_type == 'sat':
        root_dir = sat_root_dir
    elif img_type == 'osm':
        root_dir = osm_root_dir
    elif img_type == 'stv':
        root_dir = stv_root_dir
    if version.endswith("-mixed"):
        trainset_path = [trainset_path.replace('-mixed', ''), trainset_path.replace('-mixed', '-photos')]
        testset_path = [testset_path.replace('-mixed', ''), testset_path.replace('-mixed', '-photos')]
        root_dir = [root_dir.replace('-mixed', ''), root_dir.replace('-mixed', '-photos')]
    testset = OSMTextImageDataset(root_dir, testset_path, preprocessor=preprocessor)
    testloader = DataLoader(testset, batch_size=8, num_workers=4, shuffle=False)

    global_batch_size = args.batch_size
    local_batch_size = global_batch_size // world_size 

    trainset = OSMTextImageDataset(root_dir, trainset_path, preprocessor=preprocessor)
    sampler = DistributedSampler(trainset,rank=rank)
    trainloader = DataLoader(trainset, batch_size=local_batch_size, sampler=sampler,num_workers=4,pin_memory=True)
    
    if rank == 0:
        print(args)
        print("Train Image Num:", len(trainset))
        print("Test Image Num:", len(testset))
        print("Image Dir:", root_dir)
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    
    if rank == 6:
        
        print("||-------------------ZERO-SHOT--------------------||")
        
        evaluater(
                    model=model.module,
                    dataloader=testloader,
                    device=device
                )
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    
    num_epochs = args.epochs
    best_test_r_at_1 = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        dist.barrier()
        sampler.set_epoch(epoch)
            
        total_loss = 0
        r_at_1, r_at_5, r_at_10 = 0, 0, 0
        step = 0
        total_samples = len(trainloader)
        
        model.train()

        for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=(rank != 0)):

            optimizer.zero_grad()

            with autocast():
                loss = forward(model, batch, model.device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            step += 1
            
        scheduler.step()

        avg_loss = total_loss / step

        r_at_1_tensor = torch.tensor(r_at_1, dtype=torch.float32, device=device)
        r_at_5_tensor = torch.tensor(r_at_5, dtype=torch.float32, device=device)
        r_at_10_tensor = torch.tensor(r_at_10, dtype=torch.float32, device=device)
        total_loss_tensor = torch.tensor(avg_loss, dtype=torch.float32, device=device)

        dist.all_reduce(r_at_1_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(r_at_5_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(r_at_10_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

        r_at_1 = r_at_1_tensor.item()
        r_at_5 = r_at_5_tensor.item()
        r_at_10 = r_at_10_tensor.item()
        avg_loss = total_loss_tensor.item() / world_size
        
        if rank == 0:

            r_at_1 = r_at_1 / total_samples * 100
            r_at_5 = r_at_5 / total_samples * 100
            r_at_10 = r_at_10 / total_samples * 100

            
            print(f"Epoch {epoch+1}/{num_epochs}: Loss: {avg_loss:.4f}, R@1: {r_at_1:.2f}%, R@5: {r_at_5:.2f}%, R@10: {r_at_10:.2f}%")

        dist.barrier()
        """
        if (epoch + 1) % 1 == 0:
            if rank == 0:
                print("||-------------------EVALUATE--------------------||")
                r_at_1_test, _, _ = evaluater(
                    model=model.module,
                    dataloader=testloader,
                    device=device
                )
        """

    if rank == 0:
        
        os.makedirs(os.path.join(checkpoint_dir, safe_model_name), exist_ok=True)
        hyperparams = f"{args.lr}_{args.batch_size}_{args.img_type}"
        if expand:
            final_checkpoint_path = os.path.join(checkpoint_dir, safe_model_name, f'long_model_{version}_{hyperparams}_epoch{best_epoch}_{best_test_r_at_1:.2f}.pth')
        else:
            final_checkpoint_path = os.path.join(checkpoint_dir, safe_model_name, f'model_{version}_{hyperparams}_epoch{best_epoch}_{best_test_r_at_1:.2f}.pth')

        torch.save(model.module.state_dict(), final_checkpoint_path)

        print("||-------------------EVALUATE--------------------||")
        r_at_1_test, _, _ = evaluater(
            model=model.module,
            dataloader=testloader,
            device=device
        )