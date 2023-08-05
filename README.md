# robustSCL

This repo covers my experiments on applying adversarial training on supervised contrastive learning loss.

Supervised Contrastive Learning: [Paper](https://arxiv.org/abs/2004.11362)

I used this repo as an implementation for SCL loss: [Repo](https://github.com/HobbitLong/SupContrast)

## Running
You might use `CUDA_VISIBLE_DEVICES` to set the proper number of GPUs, and/or switch to CIFAR100 by `--dataset cifar100`.  
**adversarial training with Supervised Contrastive Learning**  
Pretraining stage:
```
python main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --ADV_train
```

Linear evaluation stage:
```
python main_linear_adv.py --batch_size 512 \
  --learning_rate 5 \
  --ckpt /path/to/model.pth
```







## Reference
```
@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
```
