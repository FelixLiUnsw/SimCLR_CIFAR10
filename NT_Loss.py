from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
class NT_Xent_Loss(nn.Module):
    def __init__(self):
      super(NT_Xent_Loss, self).__init__()

    # Cosine Similarity Calculation, The function has been checked, it is no issue
    def forward(self, zi, zj, temperature = 0.5):
      zi = F.normalize(zi, dim = -1)
      zj = F.normalize(zj, dim = -1)
      self.z = torch.cat([zi, zj], dim = 0)
      # if original z like this:
      # [[1.0, 0.2, 0.5, 0.7, 0.3, 0.22, 0.9],
      #  [0.2, 1.0, 0.1, 0.9, 0.2, 0.25, 0.7],
      #  [0.3, 0.4, 1.0, 0.1, 0.2, 0.03, 0.5],
      #  [0.2, 0.1, 0.2, 1.0, 0.2, 0.31, 0.7],
      #  [0.1, 0.3, 0.5, 0.7, 1.0, 0.22, 0.3],
      #  [0.2, 0.2, 0.1, 0.9, 0.2, 1.0 , 0.7],
      #  [0.3, 0.4, 0.4, 0.1, 0.2, 0.03, 1.0],
      length,_ = self.z.size()
      pairwise_similarity = torch.mm(self.z, torch.transpose(self.z, 0, 1).contiguous())
      pairwise_similarity = torch.exp(pairwise_similarity/temperature)
      # torch.eye
      # [[1,0,0,0,0,0,0],
      #  [0,1,0,0,0,0,0],
      #  [0,0,1,0,0,0,0],
      #  [0,0,0,1,0,0,0],
      #  [0,0,0,0,1,0,0],
      #  [0,0,0,0,0,1,0],
      #  [0,0,0,0,0,0,1]]
      # then, we need to convert it to true or false, 0 false, 1 true
      diagonal = torch.eye(length, device = pairwise_similarity.device).bool()
      # now, we convert 0 to true, the diagonal to false.
      diagonal = ~diagonal
      # remove some unnecessary similarity between pairs.
      # For example, pair sim(x1,x1), sim(x2,x2), sim(x3,x3) are always equal 1. But, we need to compare positive pairs and negative pairs
      # Thus, we need to remove the diagonal
      # Meanwhile, we sum up rows for denominator(all nagaive pairs)
      negative_pairs = pairwise_similarity.masked_select(diagonal).view(length, -1).sum(dim = -1)  # Dimension = 1*2N
      # now we need to do the softmax
      positive_pairs_group1 = torch.exp(torch.sum(zi* zj, dim = -1)/temperature) # loss(i,j)
      positive_pairs_group2 = torch.exp(torch.sum(zj* zi, dim = -1)/temperature) # loss(j,i)
      numerators = torch.cat([positive_pairs_group1, positive_pairs_group2], dim = 0) # 2N = (i,j) + (j,i)
      loss = ((-torch.log(numerators/negative_pairs)).sum())/length
      return loss