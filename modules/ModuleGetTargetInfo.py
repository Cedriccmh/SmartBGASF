# -*- coding: utf-8 -*-
# @Link    : https://github.com/aicezam/SmartOnmyoji
# @Version : Python3.7.6
# @MIT License Copyright (c) 2022 ACE

from os import path, walk
from re import search, compile

import numpy as np
from numpy import uint8, fromfile
# from cv2 import cv2
import cv2
from modules.ModuleImgProcess import ImgProcess
from modules.ModuleGetConfig import ReadConfigFile


class GetTargetPicOrTextInfo:
    def __init__(self, target_modname, custom_target_path, compress_val=1):
        super(GetTargetPicOrTextInfo, self).__init__()
        self.modname = target_modname
        self.custom_target_path = custom_target_path
        self.target_folder_path = None
        self.compress_val = compress_val

    def get_target_folder_path(self):
        """
        不同的模式下，匹配对应文件夹的图片
        :returns: 需要匹配的目标图片地址，如果没有返回空值
        """
        rc = ReadConfigFile()
        file_name = rc.read_config_target_path_files_name()  # 读取配置文件中的待匹配目标的名字信息

        parent_path = path.abspath(path.dirname(path.dirname(__file__)))  # 父路径

        # 通过界面上的选择目标，定位待匹配的目标文件夹
        for i in range(7):
            if self.modname == file_name[i][0]:
                target_folder_path = parent_path + r"\img\\" + file_name[i][1]
                return target_folder_path

        if self.modname == "自定义":
            target_folder_path = self.custom_target_path
            return target_folder_path
        else:
            return None

    @property
    def get_target_info(self):
        """获取目标图片文件夹路径下的所有图片信息"""
        target_img_sift = {}
        img_hw = {}
        img_name = []
        folder_path = self.get_target_folder_path()
        img_file_path = []
        cv2_img = {}
        text_data = {}

        # 获取每张图片的路径地址
        if folder_path is None:
            print("<br>未找到目标文件夹或图片地址！即将退出！")
            return None  # 脚本结束
        else:
            for cur_dir, sub_dir, included_file in walk(folder_path):
                for file in included_file:
                    full_path = path.join(cur_dir, file)
                    if search(r'\.(jpg|png)$', file):
                        img_file_path.append(full_path)
                    elif search(r'\.txt$', file):
                        text_data[file] = self.read_text_file(full_path)
                        print(f"<br>读取到文本文件: {file} <br>文本内容: {text_data[file]}")
            if not img_file_path and not text_data:
                print("未找到目标文件夹或图片地址！")
                return None

            # 通过图片地址获取每张图片的信息
            for img_path in img_file_path:
                img = cv2.imdecode(fromfile(img_path, dtype=uint8), -1)
                img_process = ImgProcess()
                img_hw[path.basename(img_path)] = img.shape[:2]
                img_name.append(self.trans_path_to_name(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                target_img_sift[path.basename(img_path)] = img_process.get_sift(img)
                cv2_img[path.basename(img_path)] = img

            return target_img_sift, img_hw, img_name, img_file_path, cv2_img, text_data  # 返回图片特征点信息，图片宽高，图片名称，图片路径地址，图片

    @staticmethod
    def trans_path_to_name(path_string):
        pattern = compile(r'([^<>/\\|:"*?]+)\.\w+$')
        return pattern.findall(path_string)[0] if pattern.findall(path_string) else None

    @staticmethod
    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file]
