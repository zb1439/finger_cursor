import numpy as np
import os.path as osp
from PIL import Image
import torch
from torchvision import models, transforms
from urllib import request

from finger_cursor.utils import queue
from .classifier import CLASSIFIER, Classifier
from .adapter import DummyAdapter, TorchAdapter


@CLASSIFIER.register()
class MobileNetV2(Classifier):
    def __init__(self, cfg):
        super(MobileNetV2, self).__init__(cfg)
        self.cls_mapping = {
            0: 2,
            1: 0,
            2: 1,
            3: 4,
            4: 5,
            5: 6,
            6: 3
        }
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        model_path = osp.join(osp.dirname(__file__), 'mobilenet.pt')
        if not osp.exists(model_path):
            print(model_path + " not found, downloading weight file from github...")
            request.urlretrieve("https://github.com/zb1439/finger_cursor/releases/download/mobilenet/mobilenet.pt",
                                model_path)
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = torch.nn.Linear(self.model.last_channel, len(self.cls_mapping))
        if torch.cuda.is_available():
            self.model.cuda()
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.adapter = DummyAdapter(cfg, None) if not cfg.MODEL.CLASSIFIER.ADAPTER.ENABLE else \
            TorchAdapter(cfg, self.model, self.cls_mapping, self.transform)

    def predict(self, features):
        feature = features["landmark"][-1]
        image = queue("frame")[-1]

        if not feature.multi_hand_landmarks:
            if self.adapter.need_adapt():
                self.adapter.no_hand_notice()
            return 0
        landmarks = feature.multi_hand_landmarks[0].landmark

        image = Image.fromarray(image[..., ::-1])
        xs = [int(l.x * image.size[0]) for l in landmarks]
        ys = [int(l.y * image.size[1]) for l in landmarks]
        boundary_x_left = max(0, min(xs) - 50)
        boundary_x_right = min(image.size[0], max(xs) + 50)
        boundary_y_up = max(0, min(ys) - 50)
        boundary_y_down = min(image.size[1], max(ys) + 50)
        image = image.crop((boundary_x_left, boundary_y_up, boundary_x_right, boundary_y_down))

        if self.adapter.need_adapt():
            self.adapter.adapt(np.array(image), feature)

        image = self.transform(image)[None]
        if torch.cuda.is_available():
            image = image.cuda()
        pred = torch.argmax(self.model(image)[0])
        if torch.cuda.is_available():
            pred = pred.cpu()
        pred = pred.numpy().item()
        return self.cls_mapping[pred]
