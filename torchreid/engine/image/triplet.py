from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss

from ..engine import Engine

from PIL import ImageDraw, ImageFont
from torchvision.utils import make_grid
from torchvision import transforms
def overlay_pid(pil_img, pid, position=(5,5), color="red", font_size=16):
    """
    在 PIL 图像上叠加 pid 信息
    """
    draw = ImageDraw.Draw(pil_img)
    try:
        # 尝试加载内置字体（或指定 TTF 字体路径）
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    text = f"PID: {pid}"
    draw.text(position, text, fill=color, font=font)
    return pil_img

class ImageTripletEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageTripletEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs)

        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary




import time

import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F


# 假设以下模块已正确导入
# from torchreid.losses import TripletLoss, CrossEntropyLoss
# import torchreid.metrics as metrics
from torchreid.utils import (
     AverageMeter, re_ranking,
    visualize_ranked_results
)





import time
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F

from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss
from torchreid.utils import AverageMeter, re_ranking, visualize_ranked_results
from ..engine import Engine


class ImageTripletEnginePose(Engine):
    r"""Engine for training with the new fusion network that takes both image and heatmap as input.
    
    主要改动：
      - parse_data_for_train: 除了 'img' 和 'pid'，还提取 'heatmap'
      - forward_backward: 调用模型时传入图像和热图，并将返回的局部特征列表拼接后计算 Triplet loss 以及交叉熵损失
      - parse_data_for_eval 和 _evaluate: 测试时同样将局部特征列表拼接后计算融合特征
    """
    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageTripletEnginePose, self).__init__(datamanager, use_gpu)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def parse_data_for_train(self, data):
        """提取训练数据中的图像、标签和热图"""
        imgs = data['img']
        pids = data['pid']
        heatmaps = data['heatmap']
        visibility= data["visibility"]
        return imgs, pids, heatmaps,visibility

    def forward_backward(self, data):
        # 提取图像、pid 和热图
        imgs, pids, heatmaps,visibility = self.parse_data_for_train(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
            heatmaps = heatmaps.float().cuda()
            visibility=visibility.float().cuda()

        # 模型前向传播，传入图像和热图，返回：
        #   x_global: 全局特征 (B, feature_dim)
        #   x_local: 局部特征列表，每个元素形状 (B, feature_dim)，共17个
        #   global_logits, local_logits: 对应分类器输出，用于交叉熵损失
        x_global, x_local, global_logits, local_logits = self.model(imgs, heatmaps,visibility)
        # 将17个局部特征拼接为 (B, 17 * feature_dim)
        # x_local_concat = torch.cat(x_local, dim=1)

        loss = 0.0
        loss_summary = {}

        # Triplet loss 部分
        if self.weight_t > 0:
            loss_t_local = self.compute_loss(self.criterion_t, x_local, pids)
            loss += 3.0 * self.weight_t * loss_t_local
            loss_summary['loss_t_local'] = loss_t_local.item()

            loss_t_global = self.compute_loss(self.criterion_t, x_global, pids)
            loss += self.weight_t * loss_t_global
            loss_summary['loss_t_global'] = loss_t_global.item()

        # Cross-entropy loss 部分
        if self.weight_x > 0:
            loss_x_global = self.compute_loss(self.criterion_x, global_logits, pids)
            loss += self.weight_x * loss_x_global
            loss_summary['loss_x_global'] = loss_x_global.item()
            loss_summary['acc_global'] = metrics.accuracy(global_logits, pids)[0].item()

            loss_x_local = self.compute_loss(self.criterion_x, local_logits, pids)
            loss += 3.0 * self.weight_x * loss_x_local
            loss_summary['loss_x_local'] = loss_x_local.item()
            loss_summary['acc_local'] = metrics.accuracy(local_logits, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary

    def parse_data_for_eval(self, data):
        """提取测试数据中的图像、pid、camid 和热图"""
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        heatmaps = data['heatmap']
        visibility = data["visibility"]
        img_path = data["img_path"]
        return imgs, pids, camids, heatmaps, visibility,img_path

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        self.set_model_mode('eval')
        batch_time = AverageMeter()

        # 修改 _feature_extraction 同时返回 impath 列表
        def _feature_extraction(data_loader):
            global_features, local_features, fusion_features = [], [], []
            pids_, camids_, impaths = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids, heatmaps, visibility, impath = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                    heatmaps = heatmaps.float().cuda()
                    visibility = visibility.float().cuda()
                end = time.time()
                # 测试时模型只返回全局特征和局部特征向量
                x_global, x_local = self.model(imgs, heatmaps, visibility)
                batch_time.update(time.time() - end)
                # 融合特征为全局与局部拼接后归一化
                fusion_feat = torch.cat([x_global, x_local], dim=1)
                fusion_feat = F.normalize(fusion_feat, p=2, dim=1)
                
                global_features.append(x_global.cpu())
                local_features.append(x_local.cpu())
                fusion_features.append(fusion_feat.cpu())

                pids_.extend(pids.tolist())
                camids_.extend(camids.tolist())
                impaths.extend(impath)   # 将 impath 加入列表
            global_features = torch.cat(global_features, 0)
            local_features = torch.cat(local_features, 0)
            fusion_features = torch.cat(fusion_features, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return global_features, local_features, fusion_features, pids_, camids_, impaths

        print('Extracting features from query set ...')
        q_global, q_local, q_fusion, q_pids, q_camids, q_impaths = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(q_fusion.size(0), q_fusion.size(1)))
        print('Extracting features from gallery set ...')
        g_global, g_local, g_fusion, g_pids, g_camids, g_impaths = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(g_fusion.size(0), g_fusion.size(1)))
        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalizing features with L2 norm ...')
            q_global = F.normalize(q_global, p=2, dim=1)
            q_local = F.normalize(q_local, p=2, dim=1)
            q_fusion = F.normalize(q_fusion, p=2, dim=1)
            g_global = F.normalize(g_global, p=2, dim=1)
            g_local = F.normalize(g_local, p=2, dim=1)
            g_fusion = F.normalize(g_fusion, p=2, dim=1)

        def evaluate_features(qf, gf, name):
            print('Computing distance matrix for {} features with metric={} ...'.format(name, dist_metric))
            distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
            if rerank:
                print('Applying person re-ranking ...')
                distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
                distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
                distmat = re_ranking(distmat, distmat_qq, distmat_gg)
            print('Computing CMC and mAP for {} features ...'.format(name))
            cmc, mAP = metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids,
                                            use_metric_cuhk03=use_metric_cuhk03)
            print('** {} Results **'.format(name))
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
            return cmc, mAP, distmat

        # 以融合特征作为示例进行 error vis（可视化错误样本）
        print('Evaluating fused features ...')
        cmc_fusion, mAP_fusion, distmat_fusion = evaluate_features(q_fusion, g_fusion, 'Fusion')
        rank1_fusion = cmc_fusion[0]

        # 错误样例可视化辅助函数
        from PIL import ImageDraw, ImageFont
        def overlay_pid(pil_img, pid, position=(5, 5), color="red", font_size=16):
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            draw.text(position, f"PID: {pid}", fill=color, font=font)
            return pil_img

        from torchvision.utils import make_grid
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        from torchreid.utils.tools import read_image as load_image
        vis_images_fusion = []  # 存储融合特征错误样例

        distmat_np_fusion = distmat_fusion.cpu().numpy()
        sorted_indices_fusion = np.argsort(distmat_np_fusion, axis=1)
        for i in range(len(q_impaths)):
            query_pid = q_pids[i]
            ranked_gallery = sorted_indices_fusion[i]
            top1_gallery_pid = g_pids[ranked_gallery[0]]
            if top1_gallery_pid != query_pid:
                query_img = load_image(q_impaths[i])
                query_img = overlay_pid(query_img, query_pid, position=(5,5), color="red", font_size=16)
                gallery_imgs = []
                for j in ranked_gallery[:5]:
                    g_img = load_image(g_impaths[j])
                    g_img = overlay_pid(g_img, g_pids[j], position=(5,5), color="red", font_size=16)
                    gallery_imgs.append(g_img)
                images_to_cat = [to_tensor(query_img)] + [to_tensor(img) for img in gallery_imgs]
                vis_grid = make_grid(images_to_cat, nrow=len(images_to_cat), padding=2)
                vis_images_fusion.append(vis_grid)
        if len(vis_images_fusion) > 0 and self.writer is not None:
            error_grid_fusion = make_grid(torch.stack(vis_images_fusion), nrow=1)
            self.writer.add_image(f'Error_Fusion_Rank1_vs_Rank1-5', error_grid_fusion, self.epoch)

        # 接下来，对局部特征也进行错误样例可视化
        print('Evaluating local features ...')
        cmc_local, mAP_local, distmat_local = evaluate_features(q_local, g_local, 'Local')
        rank1_local = cmc_local[0]
        vis_images_local = []
        distmat_np_local = distmat_local.cpu().numpy()
        sorted_indices_local = np.argsort(distmat_np_local, axis=1)
        for i in range(len(q_impaths)):
            query_pid = q_pids[i]
            ranked_gallery = sorted_indices_local[i]
            top1_gallery_pid = g_pids[ranked_gallery[0]]
            if top1_gallery_pid != query_pid:
                query_img = load_image(q_impaths[i])
                query_img = overlay_pid(query_img, query_pid, position=(5,5), color="blue", font_size=16)
                gallery_imgs = []
                for j in ranked_gallery[:5]:
                    g_img = load_image(g_impaths[j])
                    g_img = overlay_pid(g_img, g_pids[j], position=(5,5), color="blue", font_size=16)
                    gallery_imgs.append(g_img)
                images_to_cat = [to_tensor(query_img)] + [to_tensor(img) for img in gallery_imgs]
                vis_grid = make_grid(images_to_cat, nrow=len(images_to_cat), padding=2)
                vis_images_local.append(vis_grid)
        if len(vis_images_local) > 0 and self.writer is not None:
            error_grid_local = make_grid(torch.stack(vis_images_local), nrow=1)
            self.writer.add_image(f'Error_Local_Rank1_vs_Rank1-5', error_grid_local, self.epoch)

        # 也可分别评估 global 特征（此处保留）
        cmc_global, mAP_global, _ = evaluate_features(q_global, g_global, 'Global')

        return {
            'global': {'rank1': cmc_global[0], 'mAP': mAP_global},
            'local': {'rank1': rank1_local, 'mAP': mAP_local},
            'fusion': {'rank1': rank1_fusion, 'mAP': mAP_fusion}
        }
    def test(
        self,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        r"""Tests model on target datasets.
        此版本基于融合网络提取特征，并分别评估全局、局部（融合后的）以及拼接特征。
        同时，对于查询中 Rank-1 错误的样本，将 query 与 rank-1-5 的 gallery 图像拼接后输出到 TensorBoard。
        """
        self.set_model_mode('eval')
        targets = list(self.test_loader.keys())
        results = {}

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = self.test_loader[name]['query']
            gallery_loader = self.test_loader[name]['gallery']

            metrics_dict = self._evaluate(
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            print("Results for {}:".format(name))
            for key, vals in metrics_dict.items():
                print("  {} features: Rank-1: {:.1%}, mAP: {:.1%}".format(key, vals['rank1'], vals['mAP']))
            results[name] = metrics_dict

            if self.writer is not None:
                for key, vals in metrics_dict.items():
                    self.writer.add_scalar(f'Test/{name}/{key}_rank1', vals['rank1'], self.epoch)
                    self.writer.add_scalar(f'Test/{name}/{key}_mAP', vals['mAP'], self.epoch)

        return results