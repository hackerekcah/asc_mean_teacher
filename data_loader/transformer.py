import torch
# import random
import math
import numpy.random as random
from data_manager.dcase18_taskb import *
from data_manager.taskb_standrizer import *
class ToTensor(object):

    def __call__(self, sample):
        x, y = torch.from_numpy(sample[0]), torch.from_numpy(sample[1])
        x, y = x.type(torch.FloatTensor), y.type(torch.LongTensor)
        return x, y

class RandomErasing(object):
    """
    class that performs random erasing
    """
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    def __call__(self, sample):
        if random.uniform(0, 1) > self.probability:
            return sample
        img = sample[0].squeeze()
        for attempt in range(100):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                replace_number = random.uniform(-0.5, 0.5)
                img[x1: x1 + h, y1: y1 + w] = replace_number
                print(replace_number)
                return np.expand_dims(img, axis=0), sample[1]
        return sample

class Random_enhance_spl(object):
    """
    random enhance sound pressure level on spectrogram
    """
    def __init__(self):
        self.enhance_number = random.uniform(-0.2, 0.2)
    def __call__(self, sample):
        spec = sample[0]
        spec = spec.squeeze()
        spec = spec + self.enhance_number
        return np.expand_dims(spec, axis=0), sample[1]

def show_spec(img):
    plt.figure()
    librosa.display.specshow(img)
    plt.title('After random erasing')
    plt.show()

if __name__ == '__main__':
    filename = 'airport-barcelona-0-1-a.wav'
    data_manager = Dcase18TaskbData()
    data_stdrizer = TaskbStandarizer(data_manager=data_manager)
    img = data_stdrizer.load_normed_spec_by_name(filename, norm_device='a')
    sample = (np.expand_dims(img, axis=0), '3')
    # img = img.squeeze()
    data_manager.show_spec_by_name(filename)
    sample = RandomErasing(probability=0.9)(sample)
    show_spec(sample[0].squeeze())

