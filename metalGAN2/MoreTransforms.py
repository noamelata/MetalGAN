import random
import torch
import torchvision.transforms as transforms

class RandomBrightness(object):

    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, img):
        brightness = random.choice(self.brightness)
        
        transform = transforms.Compose([transforms.Lambda(
            lambda img: transforms.functional.adjust_brightness(img, brightness))])
        return transform(img)


class RandomContrast(object):

    def __init__(self, contrast):
        self.contrast = contrast

    def __call__(self, img):
        contrast = random.choice(self.contrast)

        transform = transforms.Compose([transforms.Lambda(
            lambda img: transforms.functional.adjust_contrast(img, contrast))])
        return transform(img)

