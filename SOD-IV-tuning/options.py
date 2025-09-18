import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')  # 100
parser.add_argument('--load_pre', type=str,
                    default='/xx/xx/Projects/xxxx/pretrain/eva02_L_pt_m38m_p14to16_converted.pt', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
parser.add_argument('--rgb_root', type=str,
                    default='/data2/xx/Datasets/VT5000/Train/RGB/', help='the training rgb images root')  # train_dut
parser.add_argument('--depth_root', type=str,
                    default='/data2/xx/Datasets/VT5000/Train/T/', help='the training depth images root')
parser.add_argument('--gt_root', type=str,
                    default='/data2/xx/Datasets/VT5000/Train/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str,
                    default='/data2/xx/Datasets/VT5000/Test/RGB/', help='the test gt images root')
parser.add_argument('--test_depth_root', type=str,
                    default='/data2/xx/Datasets/VT5000/Test/T/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str,
                    default='/data2/xx/Datasets/VT5000/Test/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str,
                    default='/data2/xx/Projects/CPNet-main/debug/', help='the path to save models and logs')
opt = parser.parse_args()
