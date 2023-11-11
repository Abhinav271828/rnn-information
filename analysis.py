from main import *

def analysis(e, x):
    m = load_model_from_index(e)
    h, y = m(x)
    err = y - x
    cov_e = torch.cov(err.transpose(0,1))
    _, s_e, _ = svd(cov_e.detach())

    e = sum(1 for s in s_e if s > 0.05)

    w_x = m.rnn.weight_ih_l0.detach()
    _, s_M, _ = svd(w_x.transpose(0,1).detach())
    n = len(s_M)

    ub = s_M[n-e-1]/s_M[0] if e < n else 1
    lb = s_M[n-e]/s_M[0] if e > 0 else 0
    print(f"{lb},{ub}")

if __name__ == '__main__':
    for e in range(28):
        x = OffsetData(7, 0, 1000, 1)[0][0]
        analysis(e, x)

    for e in range(28, 56):
        x = OffsetData(8, 0, 1000, 1)[0][0]
        analysis(e, x)
