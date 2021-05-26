from torchsummary import summary
from models.hrnet import HRNet
import sys
import os
import cv2
import matplotlib.pyplot as plt
import torchprof
import torch
import numpy as np
from model import SimpleHRNet
from misc.visualization import draw_points_and_skeleton, joints_dict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111, projection='3d')

ax.xaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.fill = False
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.fill = False
ax.zaxis.pane.set_edgecolor('white')
ax.grid(False)


# ax.w_zaxis.line.set_lw(0.)
# ax.set_zticks([])
# model = HRNet(32, 17, 0.1).cuda()
# y = model(torch.ones(1, 3, 384, 288).to('cuda'))
# print(y.shape)
# img = cv2.imread('frame.png')
# cv2.imshow('img', img)
# cv2.waitKey(0)
# plt.imshow(img)
# plt.show()

frame = cv2.imread('demo/input.jpg')
model = SimpleHRNet(
    48,
    17,
    'weights/w48_384_288.pth',
    model_name='HRNet',
    resolution=(384, 288),
    multiperson=False,
    return_heatmaps=True,
    return_bounding_boxes=True,
    max_batch_size=16,
    device='cuda'
)

pts = model.predict(frame)

heatmap, boxes, pts = pts
heatmap = np.squeeze(heatmap)

heatmapx = np.zeros((heatmap.shape[1], heatmap.shape[2]))
for i in range(17):
    heatmapx += heatmap[i, :, :]


heatmap = heatmap[0, :, :]


X, Y = np.meshgrid(np.linspace(0, 2, 72),
                   np.linspace(0, 2, 96))
plot = ax.plot_surface(X=X, Y=Y, Z=heatmapx, cmap='viridis')
# plt.imshow(heatmapx)
plt.show()

# print(pts)
# for i, pt in enumerate(pts):

#     frame, angle = draw_points_and_skeleton(frame, pt, joints_dict()["coco"]['skeleton'], person_index=1,
#                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
#                                             points_palette_samples=10, exercise_type=1)
# cv2.imwrite('sanvi_output.jpeg', frame)
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# plt.imshow(frame)
# plt.show()

# frame = cv2.imread('frame.png')

# plt.imshow(frame)
# plt.show()
# cv2.imshow('frame', frame)
# cv2.waitKey(0)

# start_point = (0, 0)
# end_point = (int(frame.shape[1]*0.7), int(frame.shape[0]*0.1))
# colorr = (0, 0, 0)
# thicknessr = -1
# frame = cv2.rectangle(
#     frame, start_point, end_point, colorr, thicknessr)
# font = cv2.FONT_HERSHEY_SIMPLEX
# org = (int(frame.shape[1]*0.01), int(frame.shape[0]*0.025))
# fontScale = frame.shape[0] * 0.0014
# color = (255, 255, 255)
# thickness = 1
# frame = cv2.putText(frame, 'FPS: {:.3f}'.format(122.2131), org, font,
#                     fontScale*0.35, color, thickness, cv2.LINE_AA)
# org = (int(frame.shape[1]*0.01), int(frame.shape[0]*0.08))
# thickness = 2
# text = "PushUps Count="+str(10)
# frame = cv2.putText(frame, text, org, font,
#                     fontScale, color, thickness, cv2.LINE_AA)
