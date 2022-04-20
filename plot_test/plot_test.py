import os, sys
sys.path.append('C:/Users/shchoi/Desktop/jaenananjeon_bucheo/capsnet_face_detection')

from utils import *
from model import *
from dataset import *

# def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
#     model.eval()
#     x, y = next(iter(loader))
#     x = x.to("cuda")
#     with torch.no_grad():
#         out = model(x)
#         bboxes = [[] for _ in range(x.shape[0])]
#         for i in range(3):
#             batch_size, A, S, _, _ = out[i].shape
#             anchor = anchors[i]
#             boxes_scale_i = cells_to_bboxes(
#                 out[i], anchor, S=S, is_preds=True
#             )
#             for idx, (box) in enumerate(boxes_scale_i):
#                 bboxes[idx] += box

#         model.train()

#     for i in range(batch_size):
#         nms_boxes = non_max_suppression(
#             bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
#         )
#         plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)


test_csv_path = './PASCAL_VOC/train_test.csv'
# 이거 그냥 csv path 만들면 됩니다.

test_dataset = YOLODataset(
    test_csv_path,
    transform=config.test_transforms,
    S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
    img_dir=config.IMG_DIR,
    label_dir=config.LABEL_DIR,
    anchors=config.ANCHORS,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
    shuffle=False,
    drop_last=False,
)


scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
)


plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
