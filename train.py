import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_msssim

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter, CosineScheduler, pad_img
from datasets import PairLoader
from dehazemodel import *


parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default=10,
                    type=int, help='number of workers')
parser.add_argument('--data_dir', default='RESIDE-OUT',
                    type=str, help='path to dataset')
parser.add_argument('--train_set', default='train',
                    type=str, help='train dataset name')
parser.add_argument('--val_set', default='test',
                    type=str, help='valid dataset name')
parser.add_argument('--exp', default='reside-out',
                    type=str, help='experiment setting')
parser.add_argument('--runs', default='tmp', type=str, help='')
parser.add_argument('--device', default='0', type=str, help='')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device

data_root = 'data'

torch.set_float32_matmul_precision('high')


with open(os.path.join('configs', args.exp, 'base.json'), 'r') as f:
    b_setup = json.load(f)

with open(os.path.join('configs', args.exp, 'model_'+args.model.split('_')[-1]+'.json'), 'r') as f:
    m_setup = json.load(f)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def train(train_loader, network, criterion, optimizer, scaler, cur_epoch, epoch):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in tqdm(train_loader, ncols=60):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        output = network(source_img)
        loss = criterion['loss_l2'](output, target_img)-0.5*criterion['loss_msssim'](
            target_img * 0.5 + 0.5, output * 0.5 + 0.5)+0.1*criterion['loss_CR'](output, target_img, source_img)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item())
    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    for batch in tqdm(val_loader, ncols=60):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            H, W = source_img.shape[2:]
            source_img = pad_img(source_img, network.module.patch_size if hasattr(
                network.module, 'patch_size') else 16)
            output = network(source_img)
            output = output.clamp_(-1, 1)
            output = output[:, :, :H, :W]

            ssim_val = pytorch_msssim.ssim(
                output * 0.5 + 0.5, target_img * 0.5 + 0.5, data_range=1, size_average=False).item()

        mse_loss = F.mse_loss(
            output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()

        PSNR.update(psnr.item(), source_img.size(0))
        SSIM.update(ssim_val)
    print(PSNR.avg, SSIM.avg)
    return PSNR.avg


def main():
    network = dehaze()
    network.cuda()

    # pytorch2.0
    network = torch.compile(network)  # , mode="reduce-overhead")

    # define loss function
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()
    loss_CR = ContrastLoss(args.device)
    loss_msssim = pytorch_msssim.MS_SSIM(data_range=1)

    criterion = {'loss_l1': loss_l1,
                 'loss_l2': loss_l2,
                 'loss_CR': loss_CR,
                 'loss_msssim': loss_msssim
                 }

    # define optimizer
    optimizer = torch.optim.AdamW(network.parameters(
    ), lr=m_setup['lr'], weight_decay=b_setup['weight_decay'])
    lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=b_setup['epochs'], value_min=m_setup['lr'] * 1e-2,
                                   warmup_t=b_setup['warmup_epochs'], const_t=b_setup['const_epochs'])
    scaler = GradScaler()

    # load saved model
    save_dir = os.path.join('run', args.runs, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
        best_psnr = 0
        cur_epoch = 0
    else:
        print('==> Loaded existing trained model.')
        model_info = torch.load(os.path.join(
            save_dir, args.model+'.pth'), map_location='cpu')
        network.load_state_dict(model_info['state_dict'])
        optimizer.load_state_dict(model_info['optimizer'])
        lr_scheduler.load_state_dict(model_info['lr_scheduler'])
        scaler.load_state_dict(model_info['scaler'])
        cur_epoch = model_info['cur_epoch']
        best_psnr = model_info['best_psnr']

    # define dataset
    train_dataset = PairLoader(os.path.join(data_root, args.data_dir, args.train_set),
                               'train',
                               b_setup['t_patch_size'],
                               b_setup['edge_decay'],
                               b_setup['data_augment'],
                               b_setup['cache_memory'])

    train_loader = DataLoader(train_dataset,
                              batch_size=m_setup['batch_size'],
                              sampler=RandomSampler(
                                  train_dataset, num_samples=b_setup['num_iter']),
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True)

    val_dataset = PairLoader(os.path.join(data_root, args.data_dir, args.val_set), b_setup['valid_mode'],
                             b_setup['v_patch_size'])

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)

    # start training
    print('==> Start training, current model name: ' + args.model)
    writer = SummaryWriter(
        log_dir=os.path.join('run', args.runs, args.exp))

    for epoch in range(cur_epoch, b_setup['epochs'] + 1):
        loss = train(train_loader, network, criterion, optimizer,
                     scaler, epoch, b_setup['epochs'] + 1)
        print('LOSS', loss)
        lr_scheduler.step(epoch + 1)

        writer.add_scalar('train_loss', loss, epoch)

        if epoch % b_setup['eval_freq'] == 0:
            avg_psnr = valid(val_loader, network)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({'cur_epoch': epoch + 1,
                            'best_psnr': best_psnr,
                            'state_dict': network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'scaler': scaler.state_dict()},
                           os.path.join(save_dir, args.model+'.pth'))

            writer.add_scalar('valid_psnr', avg_psnr, epoch)
            writer.add_scalar('best_psnr', best_psnr, epoch)

            print('EPOCH:', epoch, 'avg_psnr',
                  avg_psnr, 'best_psnr', best_psnr)


if __name__ == '__main__':
    main()
