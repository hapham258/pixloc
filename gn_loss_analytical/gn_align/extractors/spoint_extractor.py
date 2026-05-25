from ..dependencies.SuperGluePretrainedNetwork.models import superpoint
import numpy as np
import torch
import cv2

class SuperPointWithScale(superpoint.SuperPoint):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = superpoint.simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            superpoint.remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                superpoint.top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        return {"keypoints" : keypoints, "scores" : scores, "descriptors" : descriptors, "scale" : (0.125, 0.125)}

        
def frame2tensor(frame: np.ndarray) -> torch.Tensor:
    a = torch.tensor(frame / 255, dtype=torch.float)[None, None]
    return a

def read_img(img_name: 'str | Path', resize_size = -1) -> np.ndarray:
    img = cv2.imread(str(img_name), 0)
    h, w = img.shape[:2]
    
    if resize_size > 0:
        size = resize_size
        if size > max(h, w):
            scaler = 1.0
        else:
            scaler = size / float(max(h, w))
            img = cv2.resize(img, (int(round(w * scaler)), int(round(h * scaler))))

        return img.astype('float32'), torch.tensor([[scaler, scaler]]).to('cuda')
    
    return img.astype('float32'), torch.tensor([[1.0, 1.0]]).to('cuda')

class SuperPointExtractor:

    def __init__(self, nms_radius=4, keypoint_threshold=0.05, max_keypoints=-1, remove_borders=4, resize_size = -1):
        default_conf = {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints,
            'remove_borders': remove_borders,
            'fix_sampling': False,
        }

        self.net = SuperPointWithScale(default_conf)
        self.net.eval()
        self.net = self.net.cuda()
        self.resize_size = resize_size
    
    def extract_keypoints(self, image_path):
        image, scale = read_img(image_path, self.resize_size)

        image_shape = image.shape[-2:][::-1]
        image_shape = torch.tensor(image_shape).reshape(1, 2).cuda().float()
        image_shape = image_shape / scale

        image = frame2tensor(image).to('cuda')
        with torch.no_grad():
            pred = self.net({'image': image})
        kps = (pred['keypoints'][0] / scale) + 0.5
        scores, idx = torch.sort(pred['scores'][0], descending=True)
        kps = kps[idx]

        return kps, image_shape

    def extract_descriptors(self, image_path):
        image, scale = read_img(image_path, self.resize_size)
        image = frame2tensor(image).to('cuda')
        with torch.no_grad():
            pred = self.net({'image': image})
        scale = [pred['scale'][0] * scale[0,0], pred['scale'][1] * scale[0,1]]

        return pred['descriptors'][0].cuda(), scale

