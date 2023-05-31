import numpy as np
import torch


def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    pred=pred.squeeze(1)
    pred=pred.cpu()
    pred=pred.numpy()
    gt=gt.cpu()
    gt=gt.numpy()
    z=np.sum(gt>0)
    num = np.sum(gt == 1)
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]

# 计算 ODS 方法
def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt).to(torch.uint8)
            pred_img = ((torch.sigmoid(pred)) > thresh).to(torch.uint8)
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))
        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        # calculate precision
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        # calculate recall
        r_acc = tp / (tp + fn)
        # calculate f-score
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])
    return final_accuracy_all

# 计算 OIS 方法
def cal_ois_metrics(pred_list, gt_list, thresh_step=0.01):
    final_acc_all = []
    for pred, gt in zip(pred_list, gt_list):
        statistics = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = gt.to(torch.uint8)
            pred_img = (torch.sigmoid(pred) > thresh).to(torch.uint8)
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r_acc = tp / (tp + fn)

            if p_acc + r_acc == 0:
                f1 = 0
            else:
                f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            statistics.append([thresh, f1])
        max_f = np.nanmax(statistics, axis=0)
        final_acc_all.append(max_f[1])
    return np.nanmean(final_acc_all)
    # def get_statistics(self, pred, gt):
    #     """
    #     return tp, fp, fn
    #     """
    #     tp = np.sum((pred == 1) & (gt == 1))
    #     fp = np.sum((pred == 1) & (gt == 0))
    #     fn = np.sum((pred == 0) & (gt == 1))
    #     return [tp, fp, fn]
    #
    # def cal_prf_metrics(self, pred, gt, thresh_step=0.01):
    #     final_accuracy_all = []
    #     pred=torch.sigmoid(pred)
    #     statistics=[]
    #     for thresh in np.arange(0.0, 1.0, thresh_step):
    #         statistics = []
    #         # for pred, gt in zip(pred_list, gt_list):
    #         gt_img = (gt ).astype('uint8')
    #         pred_img = ((pred ) > thresh).astype('uint8')
    #             # calculate each image
    #         statistics.append(self.get_statistics(pred_img, gt_img))
    #     return statistics
    #         # get tp, fp, fn
    #         tp = np.sum([v[0] for v in statistics])
    #         fp = np.sum([v[1] for v in statistics])
    #         fn = np.sum([v[2] for v in statistics])
    #
    #         # calculate precision
    #         p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    #         # calculate recall
    #         r_acc = tp / (tp + fn)
    #         # calculate f-score
    #         final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])
    #     return final_accuracy_all
    #
    # # 计算 OIS 方法
    # def cal_ois_metrics(self, pred, gt, thresh_step=0.01):
    #     final_acc_all = []
    #     statistics = []
    #     pred=torch.sigmoid(pred)
    #     for thresh in np.arange(0.0, 1.0, thresh_step):
    #         gt_img = (gt ).astype('uint8')
    #         pred_img = (pred > thresh).astype('uint8')
    #         tp, fp, fn = self.get_statistics(pred_img, gt_img)
    #         p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    #         r_acc = tp / (tp + fn)
    #         if p_acc + r_acc == 0:
    #             f1 = 0
    #         else:
    #             f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
    #         statistics.append([thresh, f1])
    #     return statistics;
    #
    #     # max_f = np.amax(statistics, axis=0)
    #     # final_acc_all.append(max_f[1])
    #     # return np.mean(final_acc_all)
    #
    # # 计算 ODS 方法
    # def cal_prf_metrics(self, pred_list, gt_list, thresh_step=0.01):
    #     final_accuracy_all = []
    #     for thresh in np.arange(0.0, 1.0, thresh_step):
    #         statistics = []
    #
    #         for pred, gt in zip(pred_list, gt_list):
    #             gt_img = (gt / 255).astype('uint8')
    #             pred_img = ((pred / 255) > thresh).astype('uint8')
    #             # calculate each image
    #             statistics.append(self.get_statistics(pred_img, gt_img))
    #
    #         # get tp, fp, fn
    #         tp = np.sum([v[0] for v in statistics])
    #         fp = np.sum([v[1] for v in statistics])
    #         fn = np.sum([v[2] for v in statistics])
    #
    #         # calculate precision
    #         p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    #         # calculate recall
    #         r_acc = tp / (tp + fn)
    #         # calculate f-score
    #         final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])
    #
    #     return final_accuracy_all