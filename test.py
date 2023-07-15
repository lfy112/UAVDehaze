import os
import argparse
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
from utils import AverageMeter, write_img, chw_to_hwc, pad_img, write_img2
from datasets.loader import PairLoader
from dehazemodel import *

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', default=10,
                    type=int, help='number of workers')
parser.add_argument('--data_dir', default='', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='', type=str,
                    help='path to models saving')
parser.add_argument('--result_dir', default='', type=str,
                    help='path to results saving')
parser.add_argument('--test_set', default='test',
                    type=str, help='test dataset name')
parser.add_argument('--exp', default='reside-out',
                    type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            H, W = input.shape[2:]
            input = pad_img(input, network.patch_size if hasattr(
                network, 'patch_size') else 16)
            output, x1, x2 = network(input)
            x1 = x1.detach().cpu().squeeze(0).numpy()
            x2 = x2.detach().cpu().squeeze(0).numpy()

            for i in range(12):
                x = x1[i]
                y = x2[i]
                maxx = x.max()
                minx = x.min()
                maxy = y.max()
                miny = y.min()
                x = x-minx
                y = y-miny
                x = x*255/(maxx-minx)
                y = y*255/(maxy-miny)
                write_img2('featpic/'+filename+'swin'+str(i)+'.jpg', x)
                write_img2('featpic/'+filename+'local'+str(i)+'.jpg', y)

            output = output.clamp_(-1, 1)
            output = output[:, :, :H, :W]

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (H, W)),
                            F.adaptive_avg_pool2d(target, (H, W)),
                            data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.close()

    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, '%.03f | %.04f.csv' % (PSNR.avg, SSIM.avg)))


def main():
    network = dehaze()
    network.cuda()
    saved_model_dir = os.path.join(
        args.save_dir, args.exp, args.model+'best.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.test_set)
    test_dataset = PairLoader(dataset_dir, 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(
        args.result_dir, args.test_set, args.exp, args.model)
    test(test_loader, network, result_dir)


if __name__ == '__main__':
    main()
