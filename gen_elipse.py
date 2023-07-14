#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/23 16:02
# @Author  : apple233
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import math
import glob
import random
import os
import cv2
from img_enchance import *
import tqdm
from tqdm import trange
from multiprocessing import Process
from multiprocessing import Pool
import platform
"""
外资公章一般都为椭圆形，外资公章尺寸规定为:45X30mm，竖径尺寸为30mm、横径尺寸为45mm 转载必须保留本文地址:http://xiiie.com/html/308012.html
"""

# 离心角度求解，可得到唯一解
def cal_eccentric_angle(a, b, rad):
    return math.atan2(a * math.sin(rad), b * math.cos(rad))

def cal_ellipse_xy(a, b, degree):
    """
    根据椭圆的长短轴 角度计算椭圆上的点坐标
    ref: https://blog.csdn.net/u014789533/article/details/114692234
    Args:
        a:  x=a*cosθ   y=b*sinθ
        b:
        degree: 角度  角度θ表示原点与椭圆上一点连线与x正半轴的夹角，或称为仰角。

    Returns:
    """
    # a: 长轴
    # b: 短轴
    #
    rad = math.radians(degree) #角度 x 从度数转换为弧度。
    ecc_angle = cal_eccentric_angle(a, b, rad) #离心角度求解
    x = a * math.cos(ecc_angle)
    y = b * math.sin(ecc_angle)
    return [x, y]

def cal_draw_points( a, b, start_degree, cross_degree, split_nums=1000):
    """
    微分思想获取椭圆上的坐标点
    Args:
        a:长半轴
        b:短半轴
        start_degree:开始角度
        cross_degree:角度跨度
        split_nums:弧长切分个数 切分的越多越精准

    Returns:

    """

    result = []
    for i in range(split_nums):
        degree = start_degree + (i * cross_degree) / split_nums
        point = cal_ellipse_xy(a, b, degree)
        result.append(point)
    return result

# 欧氏距离
def get_l2_dist(point1: list, point2: list):
    return math.sqrt((point1[1] - point2[1]) ** 2 + (point1[0] - point2[0]) ** 2)


# 获取列表中所有相邻点之间的长度
def cal_points_length(xy:list):
    # xy : 坐标点列表 [[x,y],...]
    assert isinstance(xy,(list,tuple)), print("输出的xy非数组，检查类型",xy)
    total_length = 0
    part_length_list = []
    for i in range(0,len(xy)-1):
        l = get_l2_dist(xy[i],xy[i+1])
        part_length_list.append(l)
        total_length += l
    return total_length,part_length_list

# 根据起始角度、角度跨度，获取文字位置
def cal_text_pos( a, b, start_degree, cross_degree, split_nums, text_nums):
    # a: 长轴
    # b: 短轴
    # start_degree: 开始角度，与x轴正方向相同
    # cross_degree: 角度跨度
    # split_nums: 划分数量
    # text_nums: 文本长度
    xys = cal_draw_points(a, b, start_degree, cross_degree, split_nums)
    length, part_length_list = cal_points_length(xys)
    # 每个字的弧长
    text_cross_per = length / (text_nums - 1)
    # 第一个字
    result = [xys[0]]
    cross_length_cnt = 0
    for i in range(len(part_length_list)):  #
        cross_length_cnt += part_length_list[i]
        if cross_length_cnt >= text_cross_per:
            pre_dis = abs(cross_length_cnt - text_cross_per - part_length_list[i - 1])
            cur_dis = abs(cross_length_cnt - text_cross_per)
            # 取距离小的值
            if pre_dis <= cur_dis:
                result.append(xys[i])
            else:
                result.append(xys[i + 1])
            cross_length_cnt = 0
    if len(result) < text_nums:
        result.append(xys[-1])

    return result

def cal_tangent_degree(a,b,x,y):
    if y==0:
        return 90
    k = -b*b*x/(a*a*y)
    result = math.degrees(math.atan(k))
    return result

 # 计算椭圆的坐标以及角度信息
def cal_ellipse_text_info_basic(a,b,start_degree,cross_degree, split_nums,texts,top=True):
    # a: 长轴
    # b: 短轴
    # start_degree: 开始角度，与x轴正方向相同
    # cross_degree: 角度跨度
    # split_nums: 划分数量
    # text_nums: 文本长度
    result = [] # item: char, x,y degree
    text_len = len(texts)
    text_pos = cal_text_pos(a,b,start_degree,cross_degree, split_nums,text_len)
    for i in range(text_len):
        degree = cal_tangent_degree(a,b,text_pos[i][0],text_pos[i][1])
        if top and text_pos[i][1]<=0:
            degree -= 180
        if not top and text_pos[i][1]>=0:
            degree -= 180
        # y轴需要做个翻转
        result.append([texts[text_len-1-i], text_pos[i][0],text_pos[i][1],degree])
    return result, start_degree, cross_degree

# 通过对称位置，计算椭圆的坐标以及角度信息
def cal_ellipse_text_info_with_cross( a, b, cross_degree, split_nums, texts, top=True):
    # a: 长轴
    # b: 短轴
    # start_degree: 开始角度，与x轴正方向相同
    # cross_degree: 角度跨度
    # split_nums: 划分数量
    # text_nums: 文本长度
    # top: 文本在中心线上方
    start_degree = cross_degree / 2
    # print("起始角度：",start_degree)
    if not top:
        start_degree -= 180
        texts = texts[::-1]
    # 从最后一个文字开始算起点
    start_degree = 90 - start_degree
    return cal_ellipse_text_info_basic(a, b, start_degree, cross_degree, split_nums, texts, top=top)


def cal_ellipse_text_info_sim(self,a,b,font_size,space,split_nums,texts,top=True):
    """
    # 通过对称以及字体计算椭圆坐标以及角度信息，简洁版（推荐）
    Args:
    # a: 长轴
    # b: 短轴
    # font_size:  wh 文字宽高
    # space: 字间距
    # split_nums: 划分数量
    # texts: 文本
    # top : 上半区还是小半区
    Returns:
    """

    error = 1.1 # 精度，用于调整
    text_len = len(texts)
    cross_length = (font_size[0]*text_len+(text_len-1)*space)*error
    # 一半
    half_cross_length = cross_length // 2
    # 模拟从90~-90的点位置
    xys = self.cal_draw_points(a,b,270,180,split_nums)
    _, part_length_list = self.cal_points_length(xys)
    sums = 0
    cross_degree = 0
    for i in part_length_list:
        sums += i
        cross_degree += 180/split_nums
        if sums >= half_cross_length:
            break
    # 完整区域
    cross_degree *= 2
    return cal_ellipse_text_info_with_cross(a,b, cross_degree, split_nums,texts,top=top)

# 判断字符是否为中文
def is_Chinese(ch):
    if ('\u4e00' <= ch <= u'\u9fa5') or (ch == '\u3002') or(ch == '\uff1b')or(ch == '\uff0c')or(ch == '\uff1a')or(ch == '\u201c')or(ch == '\u201d')or(ch == '\uff08')or(ch == '\uff09')or(ch == '\u3001')or(ch == '\uff1f')or(ch == '\u300a')or(ch == '\u300b')or(ch == '\uff01')or(ch == '\u3010')or(ch == '\u3011')or(ch == '\uffe5'):#'\u9fff'
        return True
    return False

# 绘制旋转文字
def draw_rotated_char(image,text, font_path, pos=(150, 150), font_size=(16, 16), angle=20, color=(255, 0, 0), spacing=None,char_thickness=1):
    """
    https://www.thinbug.com/q/45179820
    Draw text at an angle into an image, takes the same arguments
    as Image.text() except for:

    :param image: Image to write text into
    :param angle: Angle to write text at
    """
    w, h = image.size
    # 求出文字占用图片大小
    max_dim = max(font_size[0]*2, font_size[1]*2)
    mask = Image.new('RGBA', (max_dim, max_dim))
    draw = ImageDraw.Draw(mask)
    # font_style = ImageFont.truetype(font=font, size=font_size[1], encoding="utf-8")
    # if is_Chinese(text):
    #     font_style = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", font_size[0], encoding="utf-8")
    # else:
    #     font_style = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size[0], encoding="utf-8")
    font_style = ImageFont.truetype(font_path, font_size[0], encoding="utf-8")
    #TODO
    char_y_ratio=font_size[1]/font_size[0]
    y_ffset = font_size[0] * (char_y_ratio - 1.0) * 0.5  # 计算y轴拉长带来的y坐标偏移量
    # font_size_y = int(char_y_ratio * font_size)
    xy = ((max_dim-font_size[1])//2,(max_dim-font_size[1])//2+int(y_ffset))

    # xy = ((max_dim-font_size[1])//2,(max_dim-font_size[1])//2)
    draw.text(xy,text,color,font=font_style,align="center",spacing=spacing,stroke_width=char_thickness) #设置位置坐标 文字 颜色 字体

    if max_dim ==font_size[1]*2:
        new_size = (int(max_dim*font_size[0]/font_size[1]),max_dim)
    else:
        new_size = (int(max_dim*font_size[0]/font_size[1]),max_dim)
    # new_size=()/
    mask = mask.resize(new_size)
    # mask = mask.rotate(angle)
    mask = mask.rotate(angle,resample=Image.BILINEAR)# 双线性  减少锯齿状
    pos = (pos[0]-mask.size[0]//2,pos[1]-mask.size[1]//2)
    image.paste(mask,pos,mask=mask)

# 通过对称以及字体计算椭圆坐标以及角度信息，简洁版（推荐）,求标签版
def cal_ellipse_text_label_info_sim(self, a, b, font_size, space, split_nums, texts, label_nums, top=True):
    # a: 长轴
    # b: 短轴
    # start_degree: 开始角度，与x轴正方向相同
    # cross_degree: 角度跨度
    # split_nums: 划分数量
    # texts: 文本
    # label_nums:  标注点数量
    # top : 上半区还是小半区
    error = 1.2  # 精度，用于调整
    text_len = len(texts)
    cross_length = (font_size[0] * text_len + text_len * space) * error
    # 一半
    half_cross_length = cross_length / 2
    # 模拟从90~-90的点位置
    xys = self.cal_draw_points(a, b, 0, 180, split_nums)
    _, part_length_list = self.cal_points_length(xys)
    sums = 0
    cross_degree = 0
    for i in part_length_list:
        sums += i
        cross_degree += 180 / split_nums
        if sums >= half_cross_length:
            break
    # 完整区域
    cross_degree *= 2
    _, start_degree, cross_degree = self.cal_ellipse_text_info_with_cross(a, b, cross_degree, split_nums, texts,
                                                                          top=top)
    texts = " " * label_nums
    return cal_ellipse_text_info_basic(a, b, start_degree, cross_degree, split_nums, texts, top=top)

# 计算某一个点绕着固定点旋转的坐标
def cal_rotate_at_fix_point(degree, valuex, valuey, center_x, center_y, clock=False):
    # degree : 旋转角度值
    # valuex, valuey : 旋转的点
    # pointx, pointy : 固定点
    # clock          : 顺时针方向
    rad = math.radians(degree) #角度转弧度
    if clock:
        rad = -rad
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex - center_x) * math.cos(rad) - (valuey - center_y) * math.sin(rad) + center_x
    nRotatey = (valuex - center_x) * math.sin(rad) + (valuey - center_y) * math.cos(rad) + center_y
    return nRotatex, nRotatey

def get_mask(img,poly):
    mask = np.zeros_like(img, dtype=np.uint8)
    # poly = np.array([[70,190],
    #             [222,190],
    #             [280,61],
    #             [330,190],
    #             [467,190],
    #             [358,260],
    #             [392,380],
    #             [280,308],
    #             [138,380],
    #             [195,260],
    #             [70,190]])

    cv2.fillPoly(mask, [poly], (255, 255, 255))
    region = mask == 0
    mask_Img = img.copy()
    mask_Img[region] = 0
    return img, mask,mask_Img

class Elipse(object):
    def __init__(self,corpus_path,backimgs_path,fonts_path,texture_path,save_path,save_path_txt,save_path_txt_2):
        """

        Args:
            corpus_path:
            backimgs_path:
            fonts_path:
            save_path:
        """

        self.corpus_path=corpus_path
        self.backimgs_path=backimgs_path
        self.fonts_path=fonts_path
        self.texture_path=texture_path
        self.save_path=save_path
        self.save_path_txt=save_path_txt
        self.save_path_txt_2=save_path_txt_2
        self.corpus_title_file = open(corpus_path, 'r', encoding='utf-8')
        self.corpus_title_text = self.corpus_title_file.readlines()
        self.corpus_360cc_file = open("./360cc_corpus.txt", 'r', encoding='utf-8')
        self.corpus_360cc_text = self.corpus_360cc_file.readlines()
        self.backimgs_list=glob.glob(backimgs_path+"/*.png")+glob.glob(backimgs_path+"/*.jpg")
        self.texture_list=glob.glob(texture_path+"/*.png")+glob.glob(texture_path+"/*.jpg")
        self.fonts_list=glob.glob(fonts_path+"/*.ttc")+glob.glob(fonts_path+"/*.ttf")
        # self.labels_file= open(self.save_path_txt, 'w+', encoding='utf-8')
        # self.labels_file_2= open(self.save_path_txt_2, 'w+', encoding='utf-8')
        self.corpus_title_file.close()
        assert len(self.backimgs_list)>0
        assert len(self.fonts_list)>0
        assert len(self.corpus_title_text)>0

        self.red_color_list=[[251,35,38   ],
                [234,109,105 ],
                [130,80,76   ],
                [190,76,67   ],
                [130,48,55   ],
                [240,84,95   ],
                [251,112,124 ],
                [207,23,39   ],
                [238,117,224 ],
                [252,115,165 ],
                [160,105,105 ],
                [148,23,21   ],
                [215,98,138  ],
                [248,155,165 ]]

        self.blue_color_list=[[130,128,211],
            [84,97,194],
            [143,142,203],
            [100,102,180],
            [150,147,229],
            [127,130,231],
            [111,127,240],
            [154,154,217],
          [242,65,255],
          [176,176,176]]

        # self.red_color_list+=self.blue_color_list

        super().__init__()


    def generate_elipseimg(self,start_end):
        """

        Args:
            corpus_path: 标题语料txt路径
            backimgs_path: 背景图路径
            fonts_path: 字体文件路径

        Returns: 带坐标和文本标注的图片

        """
        label_file = open(self.save_path_txt, 'a', encoding='utf-8')
        label_file_2 = open(self.save_path_txt_2, 'a', encoding='utf-8')
        for index in trange(start_end[0], start_end[1]):
            # back_img=Image.open(random.choice(self.backimgs_list))
            # h=random.randint(150,250)
            h=random.randint(250,300)
            # w = int(h * random.uniform(1.3, 1.6))
            type=0 #0 圆形 1 椭圆 2方形 3 三角形
            if random.random()>0.5:
                type=1
                w = int(h * random.uniform(1.3, 1.7))
                if random.random() < 0.8:
                    rotate_degree = random.randint(-20, 20)
                else:
                    rotate_degree = random.choice([90, 180, 270])

            else :
                w= h
                if random.random()<0.8:
                    rotate_degree = random.randint(-40, 40)
                else:
                    rotate_degree = random.choice([90,180,270])



            back_img=Image.new("RGBA",(w,h),(255,255,255))

            w, h = back_img.size
            # if w<h:#保证是宽大于高
            #     back_img.rotate(90,resample=Image.BILINEAR)  # 双线性  减少锯齿状)
            #     w, h = back_img.size

            # back_img=back_img.resize((w, h), Image.ANTIALIAS)
            draw = ImageDraw.Draw(back_img)

            # gen texts
            if random.random()<0.25:
                texts = random.choice(self.corpus_title_text)
            else:
                texts = random.choice(self.corpus_360cc_text)

            #纯白背景or有文字背景
            is_white_bg = True
            if random.random()<0.5:
                elipse_color = random.choice(self.blue_color_list)
            else:
                is_white_bg = False
                elipse_color = random.choice(self.red_color_list)

            font_path = random.choice(self.fonts_list)
                # font_size=int(h/random.randint(7,10))
            if len(texts)<8:
                font_size = int(h / random.randint(5, 6))
                space = font_size // random.randint(8,9)
            elif len(texts)<12:
                font_size = int(h / random.randint(6, 9))
                space = font_size // random.randint(9,11)
            elif len(texts)<15:
                font_size = int(h / (len(texts)*0.6))
                space = font_size // random.randint(12,15)
            else:
                font_size = int(h / (len(texts) * 0.6))
                # space = font_size // random.randint(20, 50)
                space = 1

            # space = 1



            x0, y0 = 0, 0
            gap = int(random.uniform(0.13,0.16)* h) #文字到圆边的距离
            shape = [x0, y0, w - x0, h - y0]
            a = w - 2 * x0
            b = h - 2 * y0
            shape2 = [x0 + gap, y0 + gap, w - x0 - gap, h - y0 - gap]
            a = w - 2 * x0 - 2 * gap #长轴
            b = h - 2 * y0 - 2 * gap #短轴
            a=a//2 #长半轴
            b=b//2#短半轴
            elipse_width = h // random.randint(25, 35)



            # elipse_color = random.choice(self.red_color_list)
            # elipse_color = self.red_color_list[0]
            # print(elipse_color)

            elipse_color = tuple(elipse_color)
            draw.ellipse(shape, outline=elipse_color, width=elipse_width)

            # texts="广电运通金融股份有限公司123"
            texts =texts.strip()
            # texts="7892584"
            texts=texts[0:18] if len(texts)>18 else texts
            # texts = "深圳市科定木饰面料科技有限公司"
            # texts = "重庆和辅广告设计有限公司"

            top = False  # w文字方向
            #内椭圆周长
            elipse_inner_length=2*math.pi*b+4*(a-b)

            # 模拟从90~-90的点位置
            split_nums = 1000
            # split_nums=800
            xys = cal_draw_points(a,b, 0, 180, split_nums)  # X横轴上1000个点的坐标 顺时针

            _, part_length_list = cal_points_length(xys)
            error = 1.05  # 精度，用于调整
            text_len = len(texts)

            for i in range(1,text_len):
                cross_length = (font_size * i + (i - 1) * space) * error
                if cross_length>0.6*elipse_inner_length: #写的文字长度不超过椭圆周长的0.7
                    text_len=i+1
                    break
            texts=texts[0:text_len]
            cross_length = (font_size * text_len + (text_len - 1) * space) * error
            # 一半
            half_cross_length = cross_length // 2
            sums = 0
            cross_degree = 0
            for i in part_length_list:
                sums += i
                cross_degree += 180 / split_nums
                if sums >= half_cross_length:
                    break
            # 完整区域
            cross_degree *= 2
            start_degree = cross_degree / 2

            if not top:
                start_degree -= 180
                texts = texts[::-1]
            # 从最后一个文字开始算起点
            start_degree = 90 - start_degree # 左右对称均匀分布
            # start_degree = random.randint(70,110) - start_degree # 左右对称均匀分布
            poly=[]
            result, start_degree, cross_degree = cal_ellipse_text_info_basic(a , b , start_degree, cross_degree,
                                                                          split_nums, texts, top=top)
            # char_thickness=random.choice([1,1,1,2])
            char_thickness=1
            # print("font_size",font_size)
            char_y_ratio=random.choice([1,1,1,1.3,1.4,1.5]) ##字体y轴拉长系数
            # char_y_ratio=1.5 #字体y轴拉长系数
            font_size_y = int(char_y_ratio * font_size)
            for (char, x, y, degree) in result:
                x_coord = int(a + x + gap)
                y_coord = int(b + y + gap)
                word_pos = (x_coord, y_coord)
                # font_size_y=int(random.choice([1,1,1,1.2,1.3,1.4,1.5])*font_size)
                draw_rotated_char(back_img, char, font_path, pos=word_pos, font_size=(font_size,font_size_y), angle=-degree,
                                  color=elipse_color,
                                  spacing=None,
                                  char_thickness=char_thickness)


                show_vis=0
                if show_vis:
                    # 计算旋转后char的坐标点
                    center_x, center_y = x_coord, y_coord
                    x0, y0 = center_x - font_size // 2, center_y - font_size // 2  # left top  顺时针
                    x1, y1 = center_x + font_size // 2, center_y - font_size // 2
                    x2, y2 = center_x + font_size // 2, center_y + font_size // 2
                    x3, y3 = center_x - font_size // 2, center_y + font_size // 2
                    x0, y0 = cal_rotate_at_fix_point(degree, x0, y0, center_x, center_y)
                    x1, y1 = cal_rotate_at_fix_point(degree, x1, y1, center_x, center_y)
                    x2, y2 = cal_rotate_at_fix_point(degree, x2, y2, center_x, center_y)
                    x3, y3 = cal_rotate_at_fix_point(degree, x3, y3, center_x, center_y)
                    draw.line([(x0,y0),(x1,y1)], fill="blue", width=3,joint="curve")
                    draw.line([(x1,y1),(x2,y2)], fill="blue", width=3,joint="curve")
                    draw.line([(x2,y2),(x3,y3)], fill="blue", width=3,joint="curve")
                    draw.line([(x3,y3),(x0,y0)], fill="blue", width=3,joint="curve")

                    draw.point((x_coord, y_coord), "red")

            #图像增强处理 先转换成opencv速度更快
            if random.random()<0.5: #印泥缺失
                img_texture=Image.open(random.choice(self.texture_list))
                back_img=texture_aug(back_img, img_texture)
            if random.random()>0.1: #压缩模糊 0.5
                back_img=compress(back_img)
            if random.random() > 0.2:  # 模糊 0.5
                back_img = blurred_aug(back_img)

            if not is_white_bg: #非纯色背景背景融合 0.2
                bground_img=Image.open(random.choice(self.backimgs_list))
                back_img=src_blend_background(back_img,bground_img)

            # return back_img

            # # 取mask区域点坐标
            # back_img=img = cv2.cvtColor(np.asarray(back_img), cv2.COLOR_RGBA2BGR)
            # poly=np.array(poly)
            # img, mask, mask_Img = get_mask(back_img, poly)

            #记录上边的点 x0,x1
            for (char, x, y, degree) in result:
                x_coord = int(a + x + gap)
                y_coord = int(b + y + gap)

                # 计算旋转后char的坐标点
                center_x, center_y = x_coord, y_coord
                point_gap=random.randint(0,font_size//5+1)
                x0, y0 = center_x - font_size // 2, center_y - font_size // 2-point_gap  # left top  顺时针
                x1, y1 = center_x + font_size // 2, center_y - font_size // 2-point_gap
                # x2, y2 = center_x + font_size // 2, center_y + font_size // 2
                # x3, y3 = center_x - font_size // 2, center_y + font_size // 2
                x0, y0 = cal_rotate_at_fix_point(degree, x0, y0, center_x, center_y)
                x1, y1 = cal_rotate_at_fix_point(degree, x1, y1, center_x, center_y)
                # x2, y2 = cal_rotate_at_fix_point(degree, x2, y2, center_x, center_y)
                # x3, y3 = cal_rotate_at_fix_point(degree, x3, y3, center_x, center_y)

                # x0, y0, x1, y1, x2, y2, x3, y3 = [int(x) for x in [x0, y0, x1, y1, x2, y2, x3, y3]]
                x0, y0, x1, y1= [int(x) for x in [x0, y0, x1, y1]]
                poly.append([x0, y0])
                poly.append([x1, y1])
                # poly.append([x2, y2])
                # poly.append([x3, y3])
            # 记录下边的点 x2,x3
            for j in range(len(result)):
                char, x, y, degree=result[len(result)-1-j]
                x_coord = int(a + x + gap)
                y_coord = int(b + y + gap)
                # 计算旋转后char的坐标点
                center_x, center_y = x_coord, y_coord
                point_gap = random.randint(0, font_size//5+1)
                # x0, y0 = center_x - font_size // 2, center_y - font_size // 2  # left top  顺时针
                # x1, y1 = center_x + font_size // 2, center_y - font_size // 2
                x2, y2 = center_x + font_size // 2, center_y + font_size // 2+point_gap
                x3, y3 = center_x - font_size // 2, center_y + font_size // 2+point_gap
                # x0, y0 = cal_rotate_at_fix_point(degree, x0, y0, center_x, center_y)
                # x1, y1 = cal_rotate_at_fix_point(degree, x1, y1, center_x, center_y)
                x2, y2 = cal_rotate_at_fix_point(degree, x2, y2, center_x, center_y)
                x3, y3 = cal_rotate_at_fix_point(degree, x3, y3, center_x, center_y)

                # x0, y0, x1, y1, x2, y2, x3, y3 = [int(x) for x in [x0, y0, x1, y1, x2, y2, x3, y3]]
                x2, y2, x3, y3 = [int(x) for x in [ x2, y2, x3, y3]]
                # poly.append([x0, y0])
                # poly.append([x1, y1])
                poly.append([x2, y2])
                poly.append([x3, y3])

            # 取mask区域点坐标
            back_img = img = cv2.cvtColor(np.asarray(back_img), cv2.COLOR_RGBA2BGR)
            poly = np.array(poly)
            img, mask, mask_Img = get_mask(back_img, poly)
            # return back_imgboundary_expand
            #对mask_Img增强

            mask_Img=Image.fromarray(np.uint8(mask_Img)) #cv2Pil

            # 随机边界扩充
            if random.random() > 0.2: #0.2
                mask_Img = boundary_expand(mask_Img)

            # 旋转
            if random.random() > 0.6:
                mask_Img = mask_Img.rotate(rotate_degree, resample=Image.BICUBIC,fillcolor="black")  # 高质量插值
            #随机缩放
            # if random.random() > 0.3:
            if random.random() > 0.3 and mask_Img.size[0]>200:
                mask_Img =resize_aug(mask_Img)

            # resize_aug
            texts=texts[::-1]
            img_name = "elipsev3_" + str(index).zfill(5) + ".png"
            mask_Img=Image.fromarray(np.uint8(mask_Img))
            save_img_path = os.path.join(save_imgspath, img_name)
            mask_Img.save(save_img_path)
            # self.labels_file.write(img_name + '\t' + texts + '\n')
            if random.random()<0.95:
                # self.labels_file.write(img_name + '\t' + texts + '\n')
                label_file.write(img_name + '\t' + texts + '\n')
            else:
                # self.labels_file_2.write(img_name + '\t' + texts + '\n')
                label_file_2.write(img_name + '\t' + texts + '\n')
            # # print("write")
            # return Image.fromarray(np.uint8(mask_Img)), texts[::-1]
            # return img, mask, mask_Img
            # return mask_Img,texts
            # text_label=img_name + '\t' + texts
            # return text_label







if __name__ == '__main__':
    import time
    corpus_path="./title_corpus.txt"
    backimgs_path="./output_background"
    fonts_path="./fonts"
    texture_path="./wenli"
    start_index = 0
    # end_index=1500000
    end_index = 300
    sys = platform.system()
    if sys == "Windows":
        save_imgspath= "./save_path/train_images"
        save_path_txt="./save_path/train_label.txt"
        save_path_txt_2="./save_path/test_label.txt"
        num_process=0
        # end_index=1500000
        end_index = 300

    else:# linux

        # #TODO 40.4服务器
        save_imgspath= '/home/wzp/project/crnn_data/general_elipse/train_images/'
        save_path_txt = '/home/wzp/project/crnn_data/general_elipse/train_label.txt'  # 训练集标签存放路径
        save_path_txt_2 = '/home/wzp/project/crnn_data/general_elipse/test_label.txt'  # 训练集标签存放路径
        end_index = 1000000
        # #A100 服务器
        # save_imgspath= '/root/project/crnn_data/general_elipse/train_images/'
        # save_path_txt = '/root/project/crnn_data/general_elipse/train_label.txt'  # 训练集标签存放路径
        # save_path_txt_2 = '/root/project/crnn_data/general_elipse/test_label.txt'  # 训练集标签存放路径

        num_process = 16  # 并行处理数据的进程数，默认1（即单进程）
    if not os.path.exists(save_imgspath):
        os.mkdir(save_imgspath)
    elipse=Elipse(corpus_path, backimgs_path, fonts_path, texture_path, save_imgspath, save_path_txt,save_path_txt_2)
    time_tic=time.time()


    if num_process <= 1:  # 单进程
        elipse.generate_elipseimg([start_index,end_index],)
        # elipse.labels_file.close()  # 关闭文件

    # 多进程
    else:
        num_samples=end_index-start_index
        data_offset=num_samples//num_process
        processes = list()
        for i in trange(start_index, end_index, data_offset):
            print(i, data_offset, num_samples)
            if i + data_offset >= num_samples:
                # self.generate_image([i, self.num_samples], )
                processes.append(Process(target=elipse.generate_elipseimg, args=([i, num_samples],)))
            else:
                processes.append(Process(target=elipse.generate_elipseimg, args=([i, i + data_offset],)))

        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # pool = Pool(num_process)
        # for i in trange(start_index, end_index, data_offset):
        #     print(i, data_offset, num_samples)
        #     if i + data_offset >= num_samples:
        #         # self.generate_image([i, self.num_samples], )
        #         # processes.append(Process(target=elipse.generate_elipseimg, args=([i, num_samples],)))
        #         pool.apply_async(func=elipse.generate_elipseimg, args=([i, num_samples],),callback=None)
        #     else:
        #         pool.apply_async(func=elipse.generate_elipseimg, args=([i, i + data_offset],),callback=None)
        #             # processes.append(Process(target=elipse.generate_elipseimg, args=([i, i + data_offset],)))
        #     pool.apply_async(func=elipse.generate_elipseimg, args=([i, num_samples],))
        # pool.close()
        # pool.join()

    # 关闭文件
    # elipse.labels_file.close()
    # elipse.labels_file_2.close()
    # labels_file.close()
    # labels_file_2.close()
    time_toc=time.time()
    print('Done! run time:', int(time_toc - time_tic) / 60)

    # # # # #
    # plt.imshow(back_img)
    # plt.show()
    # # back_img.save("111.png")
    # #

