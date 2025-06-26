python -m torch.distributed.run --nproc_per_node=4 finetune.py --lr 1e-5 --batch_size 128 --epochs 40 --version NewYork-mixed --model CLIP-L/14@336 --expand --img_type sat --logging
