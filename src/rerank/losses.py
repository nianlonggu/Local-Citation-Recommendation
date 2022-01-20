import torch
from torch import nn


class TripletLoss(nn.Module):
    def __init__(self, base_margin = 0.1, positive_irrelevance_level = 0, only_account_for_positive = True ):
        super().__init__()
        self.base_margin = base_margin      
        self.positive_irrelevance_level = positive_irrelevance_level
        self.only_account_for_positive = only_account_for_positive
    """ 
    to be able to better support multiple-GPU training, here the data mining and weighting operation
    are not done on the batch_size dimension (dimension 0), but on dimension 1 
    """
    def forward(self, sims, irrelevance_levels):
        ### sims shape:   batch_size x num_candidate
        ### irrelevance_levels shape:   batch_size x num_candidate
        ### irrelevance_levels: the lower the number is, the more relevant it is.
        
        ## treat each sim as a postive, let negative minus positive and put the results in each row
        sims_difference = sims.unsqueeze(1) - sims.unsqueeze(2)
        irrelevance_levels_difference = irrelevance_levels.unsqueeze(1) - irrelevance_levels.unsqueeze(2)
        margin = self.base_margin * irrelevance_levels_difference
        loss =  torch.clamp( sims_difference + margin, min=0)
        ## only count the loss for the positive one, (irrelevance level == 0)
        weight = torch.ones_like( loss ).masked_fill( irrelevance_levels_difference <= 0, 0.0)
        if self.only_account_for_positive:
            weight = weight.masked_fill( irrelevance_levels.unsqueeze(2) != self.positive_irrelevance_level, 0.0  )

        loss = loss*weight

        return loss.sum(2).sum(1).mean()
