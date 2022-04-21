# import os, sys
# sys.path.append('C:/Users/shchoi/Desktop/jaenananjeon_bucheo/capsnet_face_detection')

from cv2 import transform
from utils import *
from model import *
from dataset import *
from torch import optim
from PIL import Image
import torchvision.transforms as transforms

model = CapsyoloNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)

optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    
load_checkpoint('yolov3_pascal_78.1map.pth.tar', model, optimizer, config.LEARNING_RATE)

IMAGE_SIZE = 416

test_csv_path = './PASCAL_VOC/train_test.csv'



scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)

# print(scaled_anchors.size())

model.eval()

img_path = './test_image2.jpg'
img = Image.open(img_path)
tf = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE))])

x = tf(img)
x = x.unsqueeze(0).to("cuda")

with torch.no_grad():
    out = model(x)
    # [batch_size, 3, 13, 13, numclasses+5] prediction에 쓰이는 텐서 shape    
    # [batch_size, 3, 26, 26, numclasses+5] prediction에 쓰이는 텐서 shape    
    # [batch_size, 3, 52, 52, numclasses+5] prediction에 쓰이는 텐서 shape
    bboxes = [[] for _ in range(x.shape[0])]

    for i in range(3):
        # print(out[i].shape)
        # ([1, 3, 13, 13, 25])  ([1, 3, 26, 26, 25])  ([1, 3, 52, 52, 25])
        batch_size, A, S, _, _ = out[i].shape
        anchor = scaled_anchors[i]
        boxes_scale_i = cells_to_bboxes(
            out[i], anchor, S=S, is_preds=True
        )

        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box


batch_size = 1

for i in range(batch_size):
    nms_boxes = non_max_suppression(
        bboxes[i], iou_threshold=config.MAP_IOU_THRESH, threshold=config.NMS_IOU_THRESH, box_format="midpoint",
    )
    plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

