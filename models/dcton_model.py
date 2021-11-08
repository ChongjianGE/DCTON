import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import cv2
import numpy as np
import os
import ipdb
import util.util2 as util


class DCTONModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)

        self.loss_names = ['l1_clothes', 'vgg_clothes', 'l1_agnostic', 'l1_skin', 'l1_back', 'cycle', 'cycle_arm']
        self.name = opt.name

        if self.isTrain:
            self.model_names = ['G_A', 'G_EA', 'G_EB', 'G_EC', 'G_D', 'G_EA2', 'G_EB2', 'G_EC2', 'G_D2', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_EA2', 'G_EB2', 'G_EC2', 'G_D2']

        self.netG_A = networks.define_G(30, 2, opt.ngf, 'res_unet_G', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_UnetMask(4, self.gpu_ids)
        self.netG_EA, self.netG_EB, self.netG_EC, self.netG_D = networks.define_unified_G(3, 5, 3, 3, opt.ngf, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_EA2, self.netG_EB2, self.netG_EC2, self.netG_D2 = networks.define_unified_G(3, 5, 3, 3, opt.ngf, opt.init_type, opt.init_gain, self.gpu_ids)

        self.network_loading(self.netG_B, './pretrained_model/latest_net_U.pth')
        self.network_loading(self.netG_A, './pretrained_model/latest_net_G_A.pth')
        self.netG_A.eval()
        self.netG_B.eval()
        self.set_requires_grad([self.netG_A, self.netG_B], False)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = nn.Tanh()

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(3, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_C_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_D_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_E_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_F_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionPreserve = torch.nn.L1Loss()
            self.BCE = torch.nn.BCEWithLogitsLoss()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_EA.parameters(), self.netG_EB.parameters(), self.netG_EC.parameters(),
                                self.netG_D.parameters(), self.netG_EA2.parameters(), self.netG_EB2.parameters(),
                                self.netG_EC2.parameters(), self.netG_D2.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if not opt.isTrain:
            print(opt.name)
            self.network_loading(self.netG_EA2, './pretrained_model/latest_net_G_EA2.pth')
            self.network_loading(self.netG_EB2, './pretrained_model/latest_net_G_EB2.pth')
            self.network_loading(self.netG_EC2, './pretrained_model/latest_net_G_EC2.pth')
            self.network_loading(self.netG_D2, './pretrained_model/latest_net_G_D2.pth')

            self.netG_EA2.eval()
            self.netG_EB2.eval()
            self.netG_EC2.eval()
            self.netG_D2.eval()
            self.set_requires_grad([self.netG_EA2, self.netG_EB2, self.netG_EC2, self.netG_D2], False)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.img_a = input['img_a'].to(self.device)
        self.cloth_a = input['cloth_a'].to(self.device)
        self.label_a = input['label_a'].to(self.device)
        self.edge_a = input['edge_a'].to(self.device)
        self.pose_a = input['pose_a'].to(self.device)
        dense_a = input['dense_a'].to(self.device)
        self.dense_a = self.encode_input(dense_a, 25)

        label_a = self.encode_input(self.label_a, 14)
        self.arm_a = (label_a[:, 11, :, :] + label_a[:, 13, :, :]).unsqueeze(1)
        self.clothmask_a = (label_a[:, 4, :, :]).unsqueeze(1)
        self.agnostic_a = 1 - self.arm_a - self.clothmask_a
        self.back_a = (label_a[:, 0, :, :]).unsqueeze(1)
        self.low_a = (label_a[:, 5, :, :] + label_a[:, 6, :, :] + label_a[:, 8, :, :] + label_a[:, 9, :, :] + label_a[:, 10, :, :]).unsqueeze(1)
        self.hair_a = (label_a[:, 1, :, :]).unsqueeze(1)

        self.dense_back_a = self.dense_a[:, 0, :, :].unsqueeze(1)

        self.img_b = input['img_b'].to(self.device)
        self.cloth_b = input['cloth_b'].to(self.device)
        self.label_b = input['label_b'].to(self.device)
        self.edge_b = input['edge_b'].to(self.device)
        self.pose_b = input['pose_b'].to(self.device)
        dense_b = input['dense_b'].to(self.device)
        self.dense_b = self.encode_input(dense_b, 25)

        label_b = self.encode_input(self.label_b, 14)
        self.arm_b = (label_b[:, 11, :, :] + label_b[:, 13, :, :]).unsqueeze(1)
        self.clothmask_b = (label_b[:, 4, :, :]).unsqueeze(1)
        self.agnostic_b = 1 - self.arm_b - self.clothmask_a
        self.back_b = (label_b[:, 0, :, :]).unsqueeze(1)

    def encode_input(self, label, channels):
        size = label.size()
        oneHot_size = (size[0], channels, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label.data.long().cuda(), 1.0)
        return input_label

    def gen_noise(self, shape):
        noise = np.zeros(shape, dtype=np.uint8)
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()

    def max(self, mask_a, mask_b):
        return mask_a + mask_b - mask_a * mask_b

    def fushion(self, back, arm, cloth, parsing_arm, parsing_cloth):
        result = self.max(back, arm)
        result = self.max(result, cloth)
        result = self.max(result, parsing_arm)
        result = self.max(result, parsing_cloth)
        return result

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        shape = self.edge_a.shape

        ### Round 1
        GA_in = torch.cat((self.low_a, self.cloth_b, self.dense_a, self.gen_noise(shape)), 1)
        GA_out = self.netG_A(GA_in)
        self.parsing_arm_b, self.parsing_cloth_b, = self.sigmoid(GA_out[:, 0:1, :, :]), self.sigmoid(GA_out[:, 1:2, :, :])

        cloth = self.cloth_b * self.edge_b
        fake_c, warped, _, _, _, _, _, _, _ = self.netG_B(cloth, self.parsing_cloth_b, self.edge_b)
        mask = self.sigmoid(fake_c[:, 3:4, :, :])
        fake_cloth_b = self.tanh(fake_c[:, 0:3, :, :])
        self.fake_cloth_b = fake_cloth_b * (1 - mask) + mask * warped

        self.GEA_in = self.fake_cloth_b * (self.parsing_cloth_b * (1 - self.hair_a))
        self.GEB_in = torch.cat((self.img_a, self.parsing_arm_b, self.parsing_cloth_b * (1 - self.hair_a)), 1)
        self.GEC_in = self.arm_a * self.img_a
        x1 = self.netG_EA(self.GEA_in)
        x2 = self.netG_EB(self.GEB_in)
        x3 = self.netG_EC(self.GEC_in)
        fake_b = self.netG_D(x1, x2, x3)
        self.fake_b = self.tanh(fake_b)

        ### Round 2
        GA_in = torch.cat((self.low_a, self.cloth_a, self.dense_a, self.gen_noise(shape)), 1)
        GA_out = self.netG_A(GA_in)
        self.parsing_arm_a, self.parsing_cloth_a, = self.sigmoid(GA_out[:, 0:1, :, :]), self.sigmoid( GA_out[:, 1:2, :, :])

        cloth = self.cloth_a * self.edge_a
        fake_c, warped, _, _, _, _, _, _, _ = self.netG_B(cloth, self.parsing_cloth_a, self.edge_a)
        mask = self.sigmoid(fake_c[:, 3:4, :, :])
        fake_cloth_a = self.tanh(fake_c[:, 0:3, :, :])
        self.fake_cloth_a = fake_cloth_a * (1 - mask) + mask * warped

        self.GEA_in = self.fake_cloth_a * (self.parsing_cloth_a * (1 - self.hair_a))
        self.GEB_in = torch.cat((self.fake_b, self.parsing_arm_a, self.parsing_cloth_a * (1 - self.hair_a)), 1)
        self.GEC_in = self.parsing_arm_a * self.fake_b
        x1 = self.netG_EA2(self.GEA_in)
        x2 = self.netG_EB2(self.GEB_in)
        x3 = self.netG_EC2(self.GEC_in)
        fake_a = self.netG_D2(x1, x2, x3)
        self.fake_a = self.tanh(fake_a)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        pass

    def backward_D_basic(self, netD, real1, fake1, fake2):  # 1arm 2cloth
        pass

    def backward_D_A(self):

        pass

    def optimize_parameters(self, load_flag):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def forward_test(self, id):
        shape = self.edge_a.shape
        ### Round test
        GA_in = torch.cat((self.low_a, self.cloth_b, self.dense_a, self.gen_noise(shape)), 1)
        GA_out = self.netG_A(GA_in)
        self.parsing_arm_b, self.parsing_cloth_b, = self.sigmoid(GA_out[:, 0:1, :, :]), self.sigmoid(
            GA_out[:, 1:2, :, :])

        cloth = self.cloth_b * self.edge_b
        fake_c, warped, _, _, _, _, _, _, _ = self.netG_B(cloth, self.parsing_cloth_b, self.edge_a)
        mask = self.sigmoid(fake_c[:, 3:4, :, :])
        fake_cloth_a = self.tanh(fake_c[:, 0:3, :, :])
        self.fake_cloth_b = fake_cloth_a * (1 - mask) + mask * warped

        self.GEA_in = self.fake_cloth_b * (self.parsing_cloth_b * (1 - self.hair_a))
        self.GEB_in = torch.cat((self.img_a, self.parsing_arm_b, self.parsing_cloth_b * (1 - self.hair_a)), 1)
        self.GEC_in = self.parsing_arm_b * self.img_a
        x1 = self.netG_EA2(self.GEA_in)
        x2 = self.netG_EB2(self.GEB_in)
        x3 = self.netG_EC2(self.GEC_in)
        fake_b = self.netG_D2(x1, x2, x3)
        self.fake_b = self.tanh(fake_b)
        skin_a = (1 - self.dense_back_a) - (1 - self.dense_back_a) * (1 - self.back_a)

        label = self.generate_label_color((self.label_a)).float().cuda()
        # vis = [self.img_a, self.cloth_b, self.low_a, label, self.parsing_cloth_b, self.parsing_arm_b, self.fake_b]
        vis = [self.img_a, self.cloth_b, self.fake_b]
        # vis = [self.img_a, warped, fake_cloth_a, mask]
        name = self.opt.name

        self.save_pics(vis, id, name)

    def transfer_pics(self, pic_list):
        tmp_pic = []
        for pic in pic_list:
            if pic.shape[1] == 1:
                pic = torch.cat([pic, pic, pic], 1)
                tmp_pic.append(pic[0])
            else:
                pic = pic.float().cuda()
                tmp_pic.append(pic[0])
        combine = torch.cat(tmp_pic, 2).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        rgb = (cv_img * 255).astype(np.uint8)
        return rgb

    def save_pics(self, pic_list, id, name):
        tmp_pic = []
        for pic in pic_list:
            if pic.shape[1] == 1:
                pic = torch.cat([pic, pic, pic], 1)
                tmp_pic.append(pic[0])
            else:
                pic = pic.float().cuda()
                tmp_pic.append(pic[0])
        combine = torch.cat(tmp_pic, 2).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        rgb = (cv_img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        directory = './' + name + '_test/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(directory + str(id) + '.jpg', bgr)

    def network_loading(self, net, path):

        load_path = path

        if isinstance(net, torch.nn.DataParallel):
            if 'G_A' in path:
                net = net
            else:
                net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def generate_label_plain(self, inputs):
        size = inputs.size()
        pred_batch = []
        for input in inputs:
            input = input.view(1, 14, 256, 192)
            pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
            pred_batch.append(pred)

        pred_batch = np.array(pred_batch)
        pred_batch = torch.from_numpy(pred_batch)
        label_batch = pred_batch.view(size[0], 1, 256, 192)

        return label_batch

    def generate_label_color(self, inputs):
        label_batch = []
        for i in range(len(inputs)):
            label_batch.append(util.tensor2label(inputs[i], 14))
        label_batch = np.array(label_batch)
        label_batch = label_batch * 2 - 1
        input_label = torch.from_numpy(label_batch)

        return input_label



