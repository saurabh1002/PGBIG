from utils import haggling as datasets
from model import stage_4
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1

    print('>>> create models')

    net_pred = stage_4.MultiStageModel(opt=opt)
    net_pred.to(opt.cuda_idx)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        else:
            model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        # err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {})".format(ckpt['epoch']))

    print('>>> loading datasets')

    dataset = datasets.HagglingDataset(opt, split="train")
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    test_dataset = datasets.HagglingDataset(opt, split="test")
    print('>>> Test dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                pin_memory=True)

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test')
    
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))

            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))

            test_error = 0
            ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
            for j in range(1, 20):
                test_error += ret_test["#{:d}ms".format(j * 40)]

            test_error = test_error / 19
            print('testing error: {:.3f}'.format(test_error))

            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))

            if test_error < err_best:
                err_best = test_error
                is_best = True

            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                            is_best=is_best, opt=opt)

def eval(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    net_pred = stage_4.MultiStageModel(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    # load model
    model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len, map_location='cuda:0')
    net_pred.load_state_dict(ckpt['state_dict'])

    print(">>> ckpt len loaded (epoch: {})".format(ckpt['epoch']))

    dataset = datasets.HagglingDataset(opt=opt, split="test")
    data_loader = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                    pin_memory=True)

    # do test
    is_create = True
    avg_ret_log = []

    ret_test = run_model(net_pred, is_train=3, data_loader=data_loader, opt=opt)
    ret_log = np.array(['action'])
    head = np.array(['action'])

    for k in ret_test.keys():
        ret_log = np.append(ret_log, [ret_test[k]])
        head = np.append(head, ['test_' + k])

    avg_ret_log.append(ret_log[1:])
    log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
    is_create = False

    avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
    avg_ret_log = np.mean(avg_ret_log, axis=0)

    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)

def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i+1], dim=1)
    return smooth_data

def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d = 0
    else:
        titles = (np.array(range(opt.output_n)) + 1)*40
        m_p3d = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n

    st = time.time()
    for i, (p3d) in enumerate(data_loader):
        batch_size, seq_n, all_dim = p3d.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d = p3d.float().to(opt.cuda_idx)

        smooth1 = smooth(p3d,
                         sample_len=opt.input_n + opt.output_n,
                         kernel_size=opt.input_n).clone()

        smooth2 = smooth(smooth1,
                         sample_len=opt.input_n + opt.output_n,
                         kernel_size=opt.input_n).clone()

        smooth3 = smooth(smooth2,
                         sample_len=opt.input_n + opt.output_n,
                         kernel_size=opt.input_n).clone()

        input = p3d.clone()

        p3d_sup_4 = p3d.clone()[:, -out_n - in_n:].reshape(
            [-1, in_n + out_n, 19, 3])
        p3d_sup_3 = smooth1.clone()[:, -out_n - in_n:].reshape(
            [-1, in_n + out_n, 19, 3])
        p3d_sup_2 = smooth2.clone()[:, -out_n - in_n:].reshape(
            [-1, in_n + out_n, 19, 3])
        p3d_sup_1 = smooth3.clone()[:, -out_n - in_n:].reshape(
            [-1, in_n + out_n, 19, 3])

        p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1 = net_pred(input)

        p3d_out_4 = p3d_out_all_4[:, in_n:]
        p3d_out_4 = p3d_out_4.reshape([-1, out_n, all_dim//3, 3])

        p3d = p3d.reshape([-1, in_n + out_n, all_dim//3, 3])

        p3d_out_all_4 = p3d_out_all_4.reshape([batch_size, in_n + out_n, -1, 3])
        p3d_out_all_3 = p3d_out_all_3.reshape([batch_size, in_n + out_n, -1, 3])
        p3d_out_all_2 = p3d_out_all_2.reshape([batch_size, in_n + out_n, -1, 3])
        p3d_out_all_1 = p3d_out_all_1.reshape([batch_size, in_n + out_n, -1, 3])

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            loss_p3d_4 = torch.mean(torch.norm(p3d_out_all_4 - p3d_sup_4, dim=3))
            loss_p3d_3 = torch.mean(torch.norm(p3d_out_all_3 - p3d_sup_3, dim=3))
            loss_p3d_2 = torch.mean(torch.norm(p3d_out_all_2 - p3d_sup_2, dim=3))
            loss_p3d_1 = torch.mean(torch.norm(p3d_out_all_1 - p3d_sup_1, dim=3))

            loss_all = (loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1)/4
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d_4.cpu().data.numpy() * batch_size


        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d = torch.mean(torch.norm(p3d[:, in_n:in_n + out_n] - p3d_out_4, dim=3))
            m_p3d += mpjpe_p3d.cpu().data.numpy() * batch_size
        else:
            # np.save(f"output/batch{i}_gt_poses.npy", p3d.reshape(batch_size, seq_n, -1, 3).cpu().data.numpy())
            # np.save(f"output/batch{i}_pred_poses.npy", p3d_out_all_4.cpu().data.numpy())
            mpjpe_p3d = torch.sum(torch.mean(torch.norm(p3d[:, in_n:] - p3d_out_4, dim=3), dim=2), dim=0)
            m_p3d += mpjpe_p3d.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d / n
    else:
        m_p3d = m_p3d / n
        for j in range(out_n):
            ret["#{:d}ms".format(titles[j])] = m_p3d[j]
    return ret

if __name__ == '__main__':

    option = Options().parse()

    if option.is_eval == False:
        main(option)
    else:
        eval(option)