import torch
import matplotlib.pyplot as plt
import numpy as np


class Visualizer (object):
    """
    func1: plot filters as images
    func2: plot feature maps as images
    """
    def __init__(self, ckpt_file, model=None):

        self.model_state_dict = torch.load(ckpt_file)['model_state_dict']

        if model:
            self.bind_model(model)
        else:
            self.model = None

        # Tensor
        self.inputs = None
        # numpy array
        self.inspect_fmap = None
        self.inspect_fmap_name = None

        self.glance()

    def bind_model(self, model):
        """
        bind model, automatically move to cuda
        :param model:
        :return:
        """
        self.model = model
        self.model.load_state_dict(self.model_state_dict, strict=False)
        # explicit move to cuda
        self.model.cuda()
        return self

    def feed(self, x):
        """
        feed inputs and save in `self.inputs`, check Tensor device match, automatic convert if not
        :param x:
        :return:
        """
        assert self.model is not None
        if x.device == list(self.model.parameters())[0].device:
            self.inputs = x
        else:
            print("Input Tensor device not matched with model weight, auto move to cuda")
            x = x.cuda()
            self.inputs = x
            assert self.inputs.device == list(self.model.parameters())[0].device
        return self

    def glance(self):
        """
        glance at checkpoint variables and its shape
        :return:
        """
        for key, val in self.model_state_dict.items():
            print(key, val.size())

    def conv1_weight(self, name='conv1.weight', nrow=8):
        # get tensor by name
        weight_tensor = self.model_state_dict[name]

        # .copy() tp avoid manipulate original data, shape(Batch,1,H,W)
        weight = np.copy(weight_tensor.cpu().numpy())
        # shape(Batch, H, W)
        weight = np.squeeze(weight)
        vmin, vmax = weight.min(), weight.max()

        plt.figure()
        ncol = int(len(weight) / nrow)
        for i, image in enumerate(weight, start=1):
            plt.subplot(nrow, ncol, i)
            plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)

        plt.show()

    def module_out(self, module_name=None):
        # TODO: Set training == False?
        """
        given module_name, return output tensor
        :param module_name:
        :return: tensor of shape (Batch, Channels, Height, Width)
        """

        assert self.model is not None
        x = self.inputs
        for name, module in self.model.named_children():
            with torch.no_grad():
                x = module(x)
                if name == module_name:
                    return x

        print("No module matched", module_name)
        return None

    def fea_map(self, after=None):
        """
        given a module name [after], get its output feature map as `self.inspect_fmap`
        :param after: module name as a `string`
        :return: self
        """

        assert self.model is not None
        # (B, C, H, W)
        out_tensor = self.module_out(module_name=after)
        self.inspect_fmap = np.copy(out_tensor.cpu().numpy())

        self.inspect_fmap_name = after + '.out'

        print("FeatureMapSize:(B,C,H,W)=", self.inspect_fmap.shape)

        return self

    def show(self, batch_idx=None, channel_idx=None):
        """
        plot a feature map of shape (B,C,H,W), given batch index and channel index
        :param batch_idx: if None, then plot the whole batch as subplots
        :param channel_idx:
        :return: None
        """

        assert self.inspect_fmap is not None

        vmin, vmax = self.inspect_fmap.min(), self.inspect_fmap.max()

        if batch_idx is not None:
            image = self.inspect_fmap[batch_idx, channel_idx, :, :].squeeze()
            plt.imshow(image, vmin=vmin, vmax=vmax)
            title = ','.join([self.model.__class__.__name__,
                              self.inspect_fmap_name,
                              "(B:{},C:{})".format(batch_idx, channel_idx)
                              ])
            plt.title(title)
            plt.show()
        else:
            # (B, H, W)
            batch_img = self.inspect_fmap[:, channel_idx, :, :].squeeze()
            for i, image in enumerate(batch_img, start=1):
                plt.subplot(len(batch_img), 1, i)
                title = ','.join([self.model.__class__.__name__,
                                 self.inspect_fmap_name,
                                 "(B:{},C:{})".format(i - 1, channel_idx)]
                                 )
                plt.title(title)
                plt.imshow(image, vmin=vmin, vmax=vmax)
            plt.show()


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    vis = Visualizer(ckpt_file='../ckpt/Run03,BaseConv,Epoch_210,acc_0.660048.tar')

    vis.conv1_weight()

