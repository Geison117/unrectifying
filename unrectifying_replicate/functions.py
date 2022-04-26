#def update_u():
# p1
# p2
# p3
# p4
# mu1
# mu2
# mu3
# D
# DL
# v
# vl_1
# W
# b
# s
# t
import torch


def update_W(p2, c1, U, b, V, MU2):
    d, _ = b.size()
    print()
    _,J = U.size()
    I = torch.eye(d)
    sum1=0
    sum2=0
    for j in range(J):
        u = torch.unsqueeze(U[:,j], 1)
        v = torch.unsqueeze(V[:,j], 1)
        mu2 = torch.unsqueeze(MU2[:,j], 1)
        sum1 += p2*(u-b)@(v.T) + mu2@(v.T)
        sum2 += torch.inverse(p2*v@(v.T)+c1*I)
    W = sum1@sum2
    return W

def update_WL(c1, Y,b,V):
    d, _ = b.size()
    _, J = Y.size()
    I = torch.eye(d)
    sum1 = 0
    sum2 = 0
    for j in range(J):
        y = torch.unsqueeze(Y[:, j], 1)
        v = torch.unsqueeze(V[:, j], 1)
        sum1 += (y - b) @ (v.T)
        sum2 += torch.inverse( v @ (v.T) + c1 * I)
    W = sum1 @ sum2
    return W

def update_b(p2, U, W, V, Mu2):
    _, N = U.size()
    sum1 = 0
    for j in range(N):
        u = torch.unsqueeze(U[:, j], 1)
        v = torch.unsqueeze(V[:, j], 1)
        mu2 = torch.unsqueeze(Mu2[:, j], 1)

        sum1 += u - W@v + (1/p2)*mu2
    b = sum1/N
    return b

b = update_b(1, torch.rand(2,6), torch.rand(2,2), torch.rand(2,6), torch.rand(2,6))

def update_bL(Y, W, V):
    _, N = Y.size()
    sum1 = 0
    for j in range(N):
        y = torch.unsqueeze(Y[:, j], 1)
        v = torch.unsqueeze(V[:, j], 1)

        sum1 += y - W @ v
    b = sum1 / N
    return b

bL = update_bL(torch.rand(2,6), torch.rand(2,2), torch.rand(2,6))

def update_d(p1, p3, p4, c2, s, t, mu1, mu3, mu4, v, u):
    num = (p1*v + p3*s + p4*(u+t) + mu1 - mu3 + mu4)*u
    den = (1+p3+p4)*(u**2) + c2

    d = torch.clip(torch.tensor(num/den), min =0, max=1)
    return d.item()

d = update_d(0.01, 0.6, 0.02, 1000, 0.01, 0.2, 0.3, 0.5, 0.69, 0.6, 0.6)

def update_u(p1, p2, p3, p4, mu1, mu2, mu3, mu4, D, DL, v, vl_1, W, b, s, t):
    _, d = D.size()
    I = torch.eye(d)
    u = torch.inverse(p1*(D.T) @ D + p2*I + p3*(DL.T)@D + p4*(I-D).T @ (I-D)) @ (p1*(D.T) @ v \
        + p2*(W @ vl_1 + b) + p3*(D.T)@s + p4*(D-I)@t + D.T@mu1-mu2-D.T@mu3 \
        - (I-D).T@mu4)
    return u



u =  update_u(1, 2, 3, 4, torch.rand(2,1), torch.rand(2,1), torch.rand(2,1), torch.rand(2,1), torch.rand(2,2), torch.rand(2,2), torch.rand(2,1), torch.rand(2,1), torch.rand(2,2), torch.rand(2,1), torch.rand(2,1), torch.rand(2,1))
#print(u)
#print(u.size())

def update_v(p1, p2, mu1, mu2, W, u, u_l1, b, D):
    _, d = D.size()
    I = torch.eye(d)
    v = torch.inverse(p2*W.T @ W + p1*I) @ (p2*W.T @ (u_l1 - b) + p1*D@u - mu1 + W.T@mu2)
    v = torch.clip(v, min =0, max=1)
    return v

def update_vL(p1, mu1, W, y, u, b, D):
    _, d = D.size()
    I = torch.eye(d)
    v = torch.inverse(W.T @ W + p1*I) @ (W.T @ (y - b) + p1*D@u - mu1)
    v = torch.clip(v, min =0, max=1)
    return v

def update_s(p3, u, mu3, D):
    s = D@u + (mu3/p3)
    s = torch.clip(s, min =0, max=1)
    return s

def update_t(p4, u, mu4, D):
    _, d = D.size()
    I = torch.eye(d)
    t = (D-I)@u - (mu4/p4)
    t = torch.clip(t, min =0, max=1)
    return t



v = update_v(1, 2, torch.rand(2,1), torch.rand(2,1), torch.rand(2,2), torch.rand(2,1), torch.rand(2,1), torch.rand(2,1), torch.rand(2,2))
vL =update_vL(1, torch.rand(2,1), torch.rand(2,2), torch.rand(2,1), torch.rand(2,1), torch.rand(2,1), torch.rand(2,2))


def update_dual_variables(mu1, mu2, mu3, mu4, p1, p2, p3, p4, v, d, u, W, v_l1, b, t, s, i):
    mu1 = mu1 + p1*(v-d*u)
    op = u- W@v_l1 + b
    mu2 = mu2 + p2*op[i].item()
    mu3 = mu3 + p3*(d*u-s)
    mu4 = mu4 + p4*((1-d)*u + t)

    return mu1, mu2, mu3, mu4

mu1, mu2, mu3, mu4 = update_dual_variables(1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, torch.rand(2,2), torch.rand(2,1), torch.rand(2,1), 2, 1, 0)




print(d)
print(d)



#x = torch.rand(6, 5)
#print(x)
#
#y = torch.rand(5, 5)
#print(y)
#z = torch.matmul(x, y)
#
#print(z)
#print(z.size())
#
#z2 = x @ y