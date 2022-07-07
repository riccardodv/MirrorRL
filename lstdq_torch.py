import torch


def lstd_q(phis, act, r, phisp, nact, not_terminal, gamma, nb_act, add_bias=True):
    ndat = phis.shape[0]
    if add_bias:
        phis = torch.column_stack([torch.ones(ndat, 1), phis])
        phisp = torch.column_stack([torch.ones(ndat, 1), phisp])
    nfeat = phis.shape[1]

    def build_sa_feat(ps, a):
        psa = torch.zeros(ndat, nb_act, nfeat)
        act_rep = a.repeat(1, nfeat)
        return psa.scatter_(dim=1, index=act_rep.unsqueeze(1), src=ps.unsqueeze(1)).view(ndat, nfeat * nb_act)

    print('building phisa')
    phisa = build_sa_feat(phis, act)
    print('building phispa')
    phispa = build_sa_feat(phisp, nact)

    diff = phisa - gamma * phispa * not_terminal
    print('computing a')
    a = phisa.t() @ diff / ndat
    print('computing b')
    b = phisa.t() @ r / ndat
    print('calling linalg solve')
    sol = torch.linalg.lstsq(a, b, driver='gelsd')[0]
    print(f'finished. Solution norm2 {sol.pow(2).sum().sqrt()} normInf {sol.abs().max()}')
    sol = sol.view(nb_act, nfeat)
    if add_bias:
        return sol[:, 0], sol[:, 1:], phisa
    else:
        return sol, phisa
