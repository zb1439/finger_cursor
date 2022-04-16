import cv2
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as xform

from finger_cursor.utils import AdaptingException
from finger_cursor.window import imshow, notice


class Adapter:
    def __init__(self, cfg, model, cls_mapping=None, transforms=None, use_coords=False):
        self.cfg = cfg
        self.model = model
        self.classes = cfg.MODEL.CLASSIFIER.GESTURES
        self.classes[self.classes.index("n/a")] = "fist"
        self.sample_frames = cfg.MODEL.CLASSIFIER.ADAPTER.SAMPLE_FRAMES
        self.n_frames = cfg.MODEL.CLASSIFIER.ADAPTER.N_FRAMES
        self.lr = cfg.MODEL.CLASSIFIER.ADAPTER.LR
        self.epochs = cfg.MODEL.CLASSIFIER.ADAPTER.EPOCHS

        self.dump_path = os.path.join(os.getcwd(), '.adapter')
        if os.path.exists(self.dump_path):
            os.system(f"rm -rf {self.dump_path}")
        os.makedirs(self.dump_path)
        self.adapt_count = 0
        self.adapt_done = False
        self.cls_mapping = cls_mapping
        self.transform = transforms
        self.use_coords = use_coords

    def need_adapt(self):
        return not self.adapt_done

    def adapt(self, image, feature):
        raise NotImplementedError


class DummyAdapter(Adapter):
    def __init__(self, cfg, model, cls_mapping=None, transforms=None, use_coords=False):
        super(DummyAdapter, self).__init__(cfg, model)
        self.adapt_done = True

class TorchAdapter(Adapter):
    def __init__(self, cfg, model, cls_mapping=None, transforms=None, use_coords=False):
        super(TorchAdapter, self).__init__(cfg, model, cls_mapping, transforms)
        self.collected = []  # (cropped_image, coords_vector, label)
        self.current_collected = []
        if self.cls_mapping is None:
            self.cls_mapping = {i: i for i in range(len(self.classes))}
        if self.transform is None:
            self.transform = xform.Compose([
                xform.Resize((256, 256)),
                xform.ToTensor(),
            ])

    def get_label_gesture(self):
        label = self.adapt_count // self.sample_frames
        gesture = self.classes[self.cls_mapping[label]]
        return label, gesture

    def no_hand_notice(self):
        _, gesture = self.get_label_gesture()
        notice(f"show {gesture.upper()}")

    def adapt(self, image, feature):
        self.save_image(image, feature)

    def save_image(self, image, feature):
        label, gesture = self.get_label_gesture()
        if self.adapt_count % self.sample_frames == 0:
            notice(f"show {gesture.upper()}")

        imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), text=f"show {gesture.upper()}")

        if self.adapt_count % self.sample_frames >= self.sample_frames // 5:
            landmark = feature.multi_hand_landmarks[0].landmark
            coords = []
            for l in landmark:
                coords.append(l.x)
                coords.append(l.y)
                coords.append(l.z)
            coords = np.array(coords)
            self.current_collected.append((image, coords, label))

        if self.adapt_count % self.sample_frames == self.sample_frames - 1:
            sel = random.sample(self.current_collected, self.n_frames)
            self.collected.extend(sel)
            self.current_collected = []

        # count
        if self.adapt_count < len(self.classes) * self.sample_frames:
            self.adapt_count += 1
            if self.adapt_count == len(self.classes) * self.sample_frames:
                self.adapt_done = True
                notice("Training in progress...")
                self.train_model()
                notice("Training done!")

        raise AdaptingException

    def train_model(self):
        self.model.train()
        print("Adaptation started")

        def trivial_collate_fn(x):
            return x

        loader = DataLoader(self.collected, batch_size=16, shuffle=True, collate_fn=trivial_collate_fn)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        PRINT_BATCH_NUM = len(loader)

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            for i, data in enumerate(loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, coords, labels = [], [], []
                for d in data:
                    inputs.append(d[0])
                    coords.append(d[1])
                    labels.append(d[2])
                inputs = torch.tensor(np.stack([self.transform(Image.fromarray(im)) for im in inputs]), dtype=torch.float32)
                coords = torch.tensor(np.stack(coords), dtype=torch.float32)
                labels = torch.tensor(np.array(labels), dtype=torch.long)
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                    coords = coords.to('cuda')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if self.use_coords:
                    outputs = self.model(inputs, coords)
                else:
                    outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                correct += (torch.max(outputs.data, 1)[1] == labels).float().sum().item()

                # print statistics
                running_loss += loss.item()
                if i % PRINT_BATCH_NUM == (PRINT_BATCH_NUM - 1):  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_BATCH_NUM:.3f}')
                    running_loss = 0.0

        self.model.eval()

