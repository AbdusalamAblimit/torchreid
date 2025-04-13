from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
import pickle
from torchreid.utils.tools import read_image
from ..dataset import ImageDataset
import cv2
import numpy as np
class Market1501Pose(ImageDataset):
    """Market1501Pose 数据集

    基于 Market1501 数据集，在原始图像的基础上，同时加载处理出的
    关键点（保存为 _pose.pkl）和热图（保存为 _float.pkl），
    这些数据存放于数据集目录下的 pose 文件夹中，目录结构与原图一致。

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        # 允许不同的目录结构
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                '当前数据结构已弃用，请将 "bounding_box_train" 等文件夹置于 "Market-1501-v15.09.15" 下。'
            )

        # 图像数据目录
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        # pose 数据目录（存放关键点和热图）
        self.pose_dir = osp.join(self.data_dir, 'pose')

        # 检查所需文件夹是否存在
        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir, self.pose_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        # 定义对应的 pose 子目录，目录结构与图像数据保持一致
        self.train_pose_dir = osp.join(self.pose_dir, 'bounding_box_train')
        self.query_pose_dir = osp.join(self.pose_dir, 'query')
        self.gallery_pose_dir = osp.join(self.pose_dir, 'bounding_box_test')
        if self.market1501_500k:
            self.extra_gallery_pose_dir = osp.join(self.pose_dir, 'images')

        # 分别处理训练、查询和库数据，并为每个样本附加 dsetid（区分不同数据集）
        train = self.process_dir(self.train_dir, self.train_pose_dir, relabel=True, dsetid=0)
        query = self.process_dir(self.query_dir, self.query_pose_dir, relabel=False, dsetid=1)
        gallery = self.process_dir(self.gallery_dir, self.gallery_pose_dir, relabel=False, dsetid=2)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, self.extra_gallery_pose_dir, relabel=False, dsetid=2)

        super(Market1501Pose, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, img_dir, pose_dir, relabel=False, dsetid=0):
        """处理指定目录下的数据

        对于每张图像，根据文件名构造对应的关键点和热图文件路径，
        并返回 (img_path, pose_path, heatmap_path, pid, camid, dsetid) 组成的列表。
        """
        img_paths = glob.glob(osp.join(img_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # 忽略无效图片
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            # pid==0 表示背景
            assert 0 <= pid <= 1501
            # 摄像头编号范围 1~6
            assert 1 <= camid <= 6
            camid -= 1  # 将 camid 调整为从 0 开始
            if relabel:
                pid = pid2label[pid]

            # 构造对应的 pose 文件路径，文件名格式与处理代码一致
            base_name = osp.splitext(osp.basename(img_path))[0]
            pose_path = osp.join(pose_dir, f"{base_name}_pose.pkl")
            heatmap_path = osp.join(pose_dir, f"{base_name}_float.pkl")
            data.append((img_path, pid, camid, dsetid, pose_path, heatmap_path))
        return data



    def get_visibility(self, pose, thresh=0.5):
        """
        根据传入的 pose 数据（形状为 (17, 3)，最后一维为置信度），
        返回一个可见性向量，形状为 (17, 1)。
        对于每个关键点，如果置信度大于阈值，则认为可见（1），否则不可见（0）。
        """
        # 取出最后一列的置信度
        confidence = pose[:, 2]
        # 二值化：大于阈值的记为 1，否则 0
        visibility = (confidence > thresh).astype(np.float32)
        # 可将形状调整为 (17, 1) ，也可以直接返回 (17,)
        return visibility.reshape((17, 1))
    
    def __getitem__(self, index):
        """返回样本数据
        除了原始图像、pid、camid、impath 和 dsetid 外，还加载对应的关键点、热图，
        以及根据关键点计算得到的可见性。
        """
        img_path, pid, camid, dsetid, pose_path, heatmap_path = self.data[index]
        img = read_image(img_path)
        # 加载关键点数据（例如，一个 numpy 数组，形状 (17,3)）
        with open(pose_path, 'rb') as f:
            pose = pickle.load(f)
        # 加载热图数据
        with open(heatmap_path, 'rb') as f:
            heatmap = pickle.load(f)
        # heatmap 原始尺寸为 (17, 64, 48)，dtype 很可能是 float16
        new_heatmap = np.zeros((heatmap.shape[0], heatmap.shape[1], 32), dtype=heatmap.dtype)
        for i in range(heatmap.shape[0]):
            # 转换为 float32 再 resize，目标尺寸为 (width=32, height=heatmap.shape[1])
            resized = cv2.resize(heatmap[i].astype(np.float32), (32, heatmap.shape[1]), interpolation=cv2.INTER_LINEAR)
            new_heatmap[i] = resized.astype(heatmap.dtype)
        heatmap = new_heatmap
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        # 根据 pose 得到可见性信息
        visibility = self.get_visibility(pose)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid,
            'pose': pose,
            'heatmap': heatmap,
            'visibility': visibility,
            'img_path': img_path
        }
        return item
