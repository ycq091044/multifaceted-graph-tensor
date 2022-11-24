import pickle
import os
import torch
import torch.nn as nn

# load data
root = "data/"
claim_tensor = torch.Tensor(
    pickle.load(open(os.path.join(root, "claim_tensor.pkl"), "rb"))
)
county_tensor = torch.Tensor(
    pickle.load(open(os.path.join(root, "county_tensor.pkl"), "rb"))
)
covid_tensor = torch.Tensor(
    pickle.load(open(os.path.join(root, "covid_tensor.pkl"), "rb"))
)
distance_mat = torch.Tensor(
    pickle.load(open(os.path.join(root, "distance_mat.pkl"), "rb"))
)
hos_tensor = torch.Tensor(pickle.load(open(os.path.join(root, "hos_tensor.pkl"), "rb")))
mob_mat = torch.Tensor(pickle.load(open(os.path.join(root, "mob_mat.pkl"), "rb")))
vac_tensor = torch.Tensor(pickle.load(open(os.path.join(root, "vac_tensor.pkl"), "rb")))

feat_name = pickle.load(open(os.path.join(root, "feat_name.pkl"), "rb"))

# combine data
demographs = county_tensor
cty_day_feat = torch.cat(
    [claim_tensor, covid_tensor.unsqueeze(-1), hos_tensor, vac_tensor], dim=-1
)
graph1 = distance_mat
graph2 = mob_mat

print(demographs.shape, cty_day_feat.shape, graph1.shape, graph2.shape)


# build the graph tensor decomposition model
class GTD(nn.Module):
    def __init__(self, d1, d2, d3, R, alpha, beta):
        super(GTD, self).__init__()
        self.A1 = nn.Parameter(torch.rand(d1, R))
        self.A2 = nn.Parameter(torch.rand(d2, R))
        self.A3 = nn.Parameter(torch.rand(d3, R))
        self.alpha = alpha
        self.beta = beta

    def forward(self, L1, L2, T):
        # compute the loss
        rec_T = torch.einsum("ir,jr,kr->ijk", self.A1, self.A2, self.A3)
        loss1 = torch.norm(rec_T - T, p="fro") ** 2
        loss2 = torch.trace(torch.einsum("ir,ij,jm->rm", self.A1, L1, self.A1))
        loss3 = torch.trace(torch.einsum("ir,ij,jm->rm", self.A1, L2, self.A1))
        loss = loss1 + self.alpha * loss2 + self.beta * loss3
        return loss


# graph and tensor decomposition
def graph_tensor_decomposition(
    A1, A2, T, R=15, alpha=1e-1, beta=1e-1, normalized=False
):
    # get diagonal matrix
    D1 = A1.sum(axis=1)
    D2 = A2.sum(axis=1)
    if normalized:
        D_rev1 = torch.diag(D1 ** (-0.5))
        L1 = torch.diag(len(D1)) - D_rev1 @ A1 @ D_rev1
        D_rev2 = torch.diag(D2 ** (-0.5))
        L2 = torch.diag(len(D2)) - D_rev2 @ A2 @ D_rev2
    else:
        L1 = torch.diag(D1) - A1
        L2 = torch.diag(D2) - A2

    d1, d2, d3 = T.shape
    model = GTD(d1, d2, d3, R, alpha, beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    for epoch in range(50):
        loss = model(L1, L2, T)

        # loss back probagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())


graph_tensor_decomposition(
    mob_mat, distance_mat, cty_day_feat, R=5, alpha=1e-1, beta=1e-1, normalized=False
)
