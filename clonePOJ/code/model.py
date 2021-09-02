# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        targets = torch.Tensor(targets)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args,margin=0.3):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.tripletLoss = TripletLoss(margin=margin)

    def forward(self, input_ids=None, p_input_ids=None, n_input_ids=None, labels=None, multiview=None):
        bs, _ = input_ids.size()

        input_ids = torch.cat((input_ids, p_input_ids,n_input_ids), 0)

        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[1]
        #loss = self.tripletLoss(outputs,labels)
        outputs = outputs.split(bs, 0)
        #print(labels)

        prob_1 = (outputs[0] * outputs[1]).sum(-1)
        prob_2 = (outputs[0] * outputs[2]).sum(-1)
        temp = torch.cat((outputs[0], outputs[1]), 0)
        temp_labels = torch.cat((labels, labels), 0)
        prob_3 = torch.mm(outputs[0], temp.t())
        #print(prob_3)
        mask = labels[:, None] == temp_labels[None, :]
        #print(mask)
        prob_3 = prob_3 * (1 - mask.float()) - 1e9 * mask.float()

        prob = torch.softmax(torch.cat((prob_1[:, None], prob_2[:, None], prob_3), -1), -1)
        loss = torch.log(prob[:, 0] + 1e-10)
        loss = -loss.mean()


        return loss, outputs[0]



