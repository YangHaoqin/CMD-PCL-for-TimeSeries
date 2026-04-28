import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from agents.utils.soft_dtw_cuda import SoftDTW
from torch.cuda.amp import autocast, GradScaler
from agents.utils.functions import euclidean_dist, pod_loss_var, pod_loss_temp
from agents.lwf import loss_fn_kd
#import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class CMD_PCL(BaseLearner):

    def __init__(self, model, args):
        super(CMD_PCL, self).__init__(model, args)
        self.use_kd = True
        self.lambda_kd_fmap = args.lambda_kd_fmap
        self.lambda_kd_lwf = args.lambda_kd_lwf
        self.fmap_kd_metric = args.fmap_kd_metric

        self.lambda_protoAug = args.lambda_protoAug
        self.prototype = None
        self.class_label = None
        self.adaptive_weight = args.adaptive_weight
        self.num_samples = args.num_samples  # Add num_samples
        self.tau = args.tau                  # Add tau

        # PIC损失权重
        self.lambda_pic = args.lambda_pic if hasattr(args, 'lambda_pic') else 0.1

        assert args.head == 'Linear', "Currently DT2W only supports Linear single head"



    def compute_pic_loss(self, g_x, g_pos, proto_features, y):

        batch_size = g_x.size(0)

        g_x_norm = F.normalize(g_x, p=2, dim=1)
        g_pos_norm = F.normalize(g_pos, p=2, dim=1)
        proto_features_norm = F.normalize(proto_features, p=2, dim=1) if proto_features.numel() > 0 else proto_features
        

        pos_sim = torch.sum(g_x_norm * g_pos_norm, dim=1, keepdim=True) / self.tau
        

        neg_sim_proto = torch.matmul(g_x_norm, proto_features_norm.T) / self.tau if proto_features.numel() > 0 else torch.tensor([]).to(self.device)

        neg_sim_batch = torch.matmul(g_x_norm, g_x_norm.T) / self.tau
        mask = (y.unsqueeze(1) != y.unsqueeze(0)).float()
        neg_sim_batch = torch.where(mask.bool(), neg_sim_batch, torch.tensor(-1e9).to(self.device))  # 屏蔽同类样本
        

        logits = torch.cat([pos_sim, neg_sim_proto, neg_sim_batch], dim=1)
        

        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        

        return F.cross_entropy(logits, labels)

    def train_epoch(self, dataloader, epoch):

        total = 0
        correct = 0
        epoch_loss_ce = 0
        epoch_loss_kd_fmap = 0
        epoch_loss_kd_pred = 0
        epoch_loss_protoAug = 0
        epoch_loss_pic = 0
        epoch_loss = 0
        n_old_classes = self.teacher.head.out_features if self.teacher is not None else 0
        use_cuda = True if self.device == 'cuda' else False


        if self.fmap_kd_metric == 'dtw':
            similarity_metric = SoftDTW(use_cuda=use_cuda, gamma=1, normalize=False)
        elif self.fmap_kd_metric == 'euclidean':
            similarity_metric = euclidean_dist
        elif self.fmap_kd_metric == 'pod_temporal':
            similarity_metric = pod_loss_temp
        elif self.fmap_kd_metric == 'pod_variate':
            similarity_metric = pod_loss_var
        else:
            raise ValueError("Wrong metric is given!")

        scaler = GradScaler()
        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)  # x: (32, 128, 9)
            y = y.view(-1)
            total += y.size(0)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(x)  # logits: (batch_size, num_classes)
                loss_new = self.criterion(outputs, y)  # 交叉熵损失

                if self.task_now == 0:
                    # 任务0：无KD、ProtoAug或PIC损失
                    loss_kd = 0
                    loss_kd_fmap = 0
                    loss_kd_pred = 0
                    loss_protoAug = 0
                    loss_pic = 0
                else:

                    self.teacher.train()

                    with torch.no_grad():
                        teacher_logits_samples = [self.teacher(x) for _ in range(self.num_samples)]
                        teacher_probs_samples = [F.softmax(logits, dim=1) for logits in teacher_logits_samples]
                        teacher_probs_mean = torch.stack(teacher_probs_samples).mean(dim=0)
                        max_prob = teacher_probs_mean.max(dim=1)[0]
                        entropy = -torch.sum(teacher_probs_mean * torch.log(teacher_probs_mean + 1e-10), dim=1)
                        confidence = max_prob * torch.exp(-entropy)


                    student_fmap = self.model.feature_map(x)
                    teacher_fmap = self.teacher.feature_map(x)
                    if self.fmap_kd_metric == 'dtw':
                        student_fmap = student_fmap.permute(0, 2, 1)
                        teacher_fmap = teacher_fmap.permute(0, 2, 1)
                        similarity = F.cosine_similarity(student_fmap, teacher_fmap, dim=2)
                        mask = torch.sigmoid(similarity * confidence.unsqueeze(1))
                        student_fmap_masked = student_fmap * mask.unsqueeze(2)
                        teacher_fmap_masked = teacher_fmap * mask.unsqueeze(2)
                        loss_kd_fmap = similarity_metric(student_fmap_masked, teacher_fmap_masked)
                        loss_kd_fmap = torch.mean(loss_kd_fmap)
                    else:
                        loss_kd_fmap = similarity_metric(student_fmap, teacher_fmap)
                        loss_kd_fmap = torch.mean(loss_kd_fmap)


                    loss_kd_pred = 0
                    if self.lambda_kd_lwf > 0:
                        cur_model_logits = outputs[:, :self.teacher.head.out_features]
                        with torch.no_grad():
                            teacher_logits = self.teacher(x)
                        loss_kd_pred = loss_fn_kd(cur_model_logits, teacher_logits)

                    loss_kd = self.lambda_kd_fmap * loss_kd_fmap + self.lambda_kd_lwf * loss_kd_pred


                    loss_protoAug = 0
                    if self.lambda_protoAug > 0:
                        proto_aug = []
                        proto_aug_label = []
                        index = list(range(n_old_classes))
                        for _ in range(self.args.batch_size):
                            np.random.shuffle(index)
                            temp = self.prototype[index[0]] + np.random.normal(0, 1, self.args.feature_dim) * self.radius
                            proto_aug.append(temp)
                            proto_aug_label.append(self.class_label[index[0]])
                        proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
                        proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
                        soft_feat_aug = self.model.head(proto_aug)
                        loss_protoAug = self.criterion(soft_feat_aug, proto_aug_label)
                        loss_protoAug = self.lambda_protoAug * loss_protoAug


                    loss_pic = 0
                    if self.lambda_pic > 0:
                        g_x = self.model.feature(x)
                        x_pos = []
                        for i in range(y.size(0)):
                            same_class_indices = (y == y[i]).nonzero(as_tuple=True)[0]
                            if same_class_indices.size(0) > 1:
                                pos_idx = same_class_indices[same_class_indices != i]
                                pos_idx = pos_idx[torch.randint(0, pos_idx.size(0), (1,))].item()
                            else:
                                pos_idx = i
                            x_pos.append(x[pos_idx])
                        x_pos = torch.stack(x_pos, dim=0)
                        g_pos = self.model.feature(x_pos)

                        if self.prototype is not None:
                            proto_features = torch.tensor(self.prototype, dtype=torch.float32).to(self.device)
                        else:
                            proto_features = torch.empty(0, g_x.size(1)).to(self.device)

                        loss_pic = self.compute_pic_loss(g_x, g_pos, proto_features, y)


                if self.adaptive_weight:
                    step_loss = (1 / (self.task_now + 1)) * loss_new + (1 - 1 / (self.task_now + 1)) * (
                                loss_kd + loss_protoAug + self.lambda_pic * loss_pic)
                else:
                    step_loss = loss_new + loss_kd + loss_protoAug + self.lambda_pic * loss_pic


            scaler.scale(step_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()


            epoch_loss += step_loss.item()
            epoch_loss_ce += loss_new.item()
            epoch_loss_kd_fmap += loss_kd_fmap.item() if self.task_now > 0 else 0
            epoch_loss_kd_pred += loss_kd_pred.item() if self.task_now > 0 else 0
            epoch_loss_protoAug += loss_protoAug.item() if self.task_now > 0 else 0
            epoch_loss_pic += loss_pic.item() if self.task_now > 0 else 0
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()


        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)
        epoch_loss_ce /= (batch_id + 1)
        epoch_loss_kd_fmap /= (batch_id + 1)
        epoch_loss_kd_pred /= (batch_id + 1)
        epoch_loss_protoAug /= (batch_id + 1)
        epoch_loss_pic /= (batch_id + 1)


        return (epoch_loss, epoch_loss_ce, epoch_loss_kd_fmap, epoch_loss_kd_pred, epoch_loss_protoAug, epoch_loss_pic), epoch_acc

    def epoch_loss_printer(self, epoch, acc, loss):

        print('Epoch {}/{}: Accuracy = {:.2f}%, Total_loss = {:.4f}, '
              'CE = {:.4f}, DT2W = {:.4f}, LwF = {:.4f}, protoAug_loss = {:.4f}, PIC_loss = {:.4f}'.format(
                  epoch + 1, self.epochs, acc, loss[0], loss[1],
                  self.lambda_kd_fmap * loss[2], self.lambda_kd_lwf * loss[3], loss[4], loss[5]))

    def after_task(self, x_train, y_train):
        super(DT2W, self).after_task(x_train, y_train)
        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=False)
        if self.lambda_protoAug > 0:
            self.protoSave(model=self.model, loader=dataloader, current_task=self.task_now)


    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                feature = model.feature(x.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(y.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = np.array(prototype)
            self.class_label = np.array(class_label)
            print(self.radius)
        else:
            new_prototypes = np.array(prototype)
            self.prototype = np.vstack((new_prototypes, self.prototype))
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
        model.train()

