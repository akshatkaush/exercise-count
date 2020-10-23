import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import ffmpeg
import math


def joints_dict():
    joints = {
        "coco": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
            },
            "skeleton": [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
                [0, 5], [0, 6]
            ]
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist"
            },
            "skeleton": [
                # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
            ]
        },
    }
    return joints

def draw_points_chinups(image, points, color_palette='tab20', palette_samples=16, confidence_threshold=0.5):
    
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))
    count=0
    y1=0
    y2=0
    y3=0
    ylw=points[9][0]
    yrw=points[10][0]
    z1,z2,z3=0,0,0
    
    for i, pt in enumerate(points):
        if pt[2] > confidence_threshold:
            image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[i % len(colors)]), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            org = (pt[1], pt[0])
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            image = cv2.putText(image, str(i) , org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
           
        if i==0:
            #x1=pt[1]
            y1=pt[0] 
            z1=pt[2]
            #print(x1," ",y1)
        if i==1:
            #x2=pt[1]
            y2=pt[0]
            z2=pt[2]
            #print(x2," ",y2) 
        if i==2:
            #x3=pt[1]
            y3=pt[0]
            z3=pt[2]
            #print(x3," ",y3)
            
        dist=distance(y1,y2,y3,z1,z2,z3,ylw,yrw)            
            
    return image,dist

def draw_points_situps(image, points, color_palette='tab20', palette_samples=16, confidence_threshold=0.5):
    
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))
    count=0
    x1=0
    x2=0
    y1=0
    y2=0
    x3=0
    y3=0
    xn=points[0][1]
    xlh=points[11][1]
    xrh=points[12][1]
    if(xn<xlh or xn<xrh):
        a,b,c=5,7,9
        print("left")
    else:
        a,b,c=6,8,10
        print("right")
    #print("here")
        
    
    for i, pt in enumerate(points):
        if pt[2] > confidence_threshold:
            image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[i % len(colors)]), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            org = (pt[1], pt[0])
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            image = cv2.putText(image, str(i) , org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
           
        if i==11:
            x1=pt[1]
            y1=pt[0] 
            #print(x1," ",y1)
        if i==13:
            x2=pt[1]
            y2=pt[0]
            #print(x2," ",y2) 
        if i==15:
            x3=pt[1]
            y3=pt[0]
            #print(x3," ",y3)
            
        ang=angle(x1,y1,x2,y2,x3,y3)            
            
    return image,ang

def draw_points_pushups(image, points, color_palette='tab20', palette_samples=16, confidence_threshold=0.5):
    
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))
    count=0
    x1=0
    x2=0
    y1=0
    y2=0
    x3=0
    y3=0
    xn=points[0][1]
    xlh=points[11][1]
    xrh=points[12][1]
    if(xn<xlh or xn<xrh):
        a,b,c=5,7,9
        print("left")
    else:
        a,b,c=6,8,10
        print("right")
    #print("here")
        
    
    for i, pt in enumerate(points):
        if pt[2] > confidence_threshold:
            image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[i % len(colors)]), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            org = (pt[1], pt[0])
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            image = cv2.putText(image, str(i) , org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
           
        if i==a:
            x1=pt[1]
            y1=pt[0] 
            #print(x1," ",y1)
        if i==b:
            x2=pt[1]
            y2=pt[0]
            #print(x2," ",y2) 
        if i==c:
            x3=pt[1]
            y3=pt[0]
            #print(x3," ",y3)
            
        ang=angle(x1,y1,x2,y2,x3,y3)            
            
    return image,ang


def draw_skeleton(image, points, skeleton, color_palette='Set2', palette_samples=8, person_index=0,
                  confidence_threshold=0.5):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    for i, joint in enumerate(skeleton):
        pt1, pt2 = points[joint]
        if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
            image = cv2.line(
                image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                tuple(colors[person_index % len(colors)]), 2
            )

    return image


def draw_points_and_skeleton(image, points, skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5, exercise_type=1):
    
    image = draw_skeleton(image, points, skeleton, color_palette=skeleton_color_palette,
                          palette_samples=skeleton_palette_samples, person_index=person_index,
                          confidence_threshold=confidence_threshold)
    if exercise_type==1:
        image,angle = draw_points_pushups(image, points, color_palette=points_color_palette, palette_samples=points_palette_samples,
                            confidence_threshold=confidence_threshold)
    if exercise_type==2:
        image,angle = draw_points_situps(image, points, color_palette=points_color_palette, palette_samples=points_palette_samples,
                        confidence_threshold=confidence_threshold)
    if exercise_type==3:
        image,angle = draw_points_chinups(image, points, color_palette=points_color_palette, palette_samples=points_palette_samples,
                        confidence_threshold=confidence_threshold)
        
    return image,angle


def save_images(images, target, joint_target, output, joint_output, joint_visibility, summary_writer=None, step=0,
                prefix=''):
    """
    Creates a grid of images with gt joints and a grid with predicted joints.
    This is a basic function for debugging purposes only.

    If summary_writer is not None, the grid will be written in that SummaryWriter with name "{prefix}_images" and
    "{prefix}_predictions".

    Args:
        images (torch.Tensor): a tensor of images with shape (batch x channels x height x width).
        target (torch.Tensor): a tensor of gt heatmaps with shape (batch x channels x height x width).
        joint_target (torch.Tensor): a tensor of gt joints with shape (batch x joints x 2).
        output (torch.Tensor): a tensor of predicted heatmaps with shape (batch x channels x height x width).
        joint_output (torch.Tensor): a tensor of predicted joints with shape (batch x joints x 2).
        joint_visibility (torch.Tensor): a tensor of joint visibility with shape (batch x joints).
        summary_writer (tb.SummaryWriter): a SummaryWriter where write the grids.
            Default: None
        step (int): summary_writer step.
            Default: 0
        prefix (str): summary_writer name prefix.
            Default: ""

    Returns:
        A pair of images which are built from torchvision.utils.make_grid
    """
    # Input images with gt
    images_ok = images.detach().clone()
    images_ok[:, 0].mul_(0.229).add_(0.485)
    images_ok[:, 1].mul_(0.224).add_(0.456)
    images_ok[:, 2].mul_(0.225).add_(0.406)
    for i in range(images.shape[0]):
        joints = joint_target[i] * 4.
        joints_vis = joint_visibility[i]

        for joint, joint_vis in zip(joints, joints_vis):
            if joint_vis[0]:
                a = int(joint[1].item())
                b = int(joint[0].item())
                # images_ok[i][:, a-1:a+1, b-1:b+1] = torch.tensor([1, 0, 0])
                images_ok[i][0, a - 1:a + 1, b - 1:b + 1] = 1
                images_ok[i][1:, a - 1:a + 1, b - 1:b + 1] = 0
    grid_gt = torchvision.utils.make_grid(images_ok, nrow=int(images_ok.shape[0] ** 0.5), padding=2, normalize=False)
    if summary_writer is not None:
        summary_writer.add_image(prefix + 'images', grid_gt, global_step=step)

    # Input images with prediction
    images_ok = images.detach().clone()
    images_ok[:, 0].mul_(0.229).add_(0.485)
    images_ok[:, 1].mul_(0.224).add_(0.456)
    images_ok[:, 2].mul_(0.225).add_(0.406)
    for i in range(images.shape[0]):
        joints = joint_output[i] * 4.
        joints_vis = joint_visibility[i]

        for joint, joint_vis in zip(joints, joints_vis):
            if joint_vis[0]:
                a = int(joint[1].item())
                b = int(joint[0].item())
                # images_ok[i][:, a-1:a+1, b-1:b+1] = torch.tensor([1, 0, 0])
                images_ok[i][0, a - 1:a + 1, b - 1:b + 1] = 1
                images_ok[i][1:, a - 1:a + 1, b - 1:b + 1] = 0
    grid_pred = torchvision.utils.make_grid(images_ok, nrow=int(images_ok.shape[0] ** 0.5), padding=2, normalize=False)
    if summary_writer is not None:
        summary_writer.add_image(prefix + 'predictions', grid_pred, global_step=step)

    return grid_gt, grid_pred
   

def angle(x1,y1,x2,y2,x3,y3):
    a=math.sqrt((x3-x2)**2+(y3-y2)**2)
    b=math.sqrt((x3-x1)**2+(y3-y1)**2)
    c=math.sqrt((x2-x1)**2+(y2-y1)**2)
    if a==0 or c==0:
        angle=0
        return angle
    else:
        term=(a**2+c**2-b**2)/(2*c*a)
        angle_rad=math.acos(term)
        angle=(180*angle_rad)/(math.pi)
    return angle

def distance(y1,y2,y3,z1,z2,z3,ylw,yrw):
    t1,t2,t3=0,0,0
    if(z1>0.5 and y1>ylw and y1>yrw):
        t1=1
    if(z2>0.5 and y2>ylw and y2>yrw):
        t2=1
    if(z3>0.5 and y3>ylw and y3>yrw):
        t3=1
        
    if(t1==1 and t2==1):
        return 1
    if(t1==1 and t3==1):
        return 1
    if(t1==2 and t3==1):
        return 1
    
    return -1
