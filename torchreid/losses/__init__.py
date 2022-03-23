from __future__ import division, print_function, absolute_import

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss


def DeepSupervision(criterion, xs, y):  #y.shape = bs * seqlen
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss

def DeepSupervision_kspattention(criterion, xs, y,f_sum):
    loss = 0.
    loss_amount = len(xs)
    if loss_amount == 4:
        v_g_ratio = 1
    else:
        v_g_ratio = 4 / 6
    for idx, x in enumerate(xs):
        if idx <= 3:
            #for i in range(x.shape[0]):
                #x[i, :] = x[i, :] * f_sum[i, idx]
            x = ((x.permute(1,0).cuda() * f_sum[:, idx].cuda() * (1 / v_g_ratio)).permute(1,0)).cuda() # 更简单
            #x = x.cuda()

        loss += criterion(x, y)
    loss /= len(xs)
    return loss