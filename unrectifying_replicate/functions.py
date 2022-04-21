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
    #print(v)
    v = torch.clip(v, min =0, max=1)
    return v

def update_vL(p1, mu1, W, y, u, b, D):
    _, d = D.size()
    I = torch.eye(d)
    v = torch.inverse(W.T @ W + p1*I) @ (W.T @ (y - b) + p1*D@u - mu1)
    v = torch.clip(v, min =0, max=1)
    return v

v = update_v(1, 2, torch.rand(2,1), torch.rand(2,1), torch.rand(2,2), torch.rand(2,1), torch.rand(2,1), torch.rand(2,1), torch.rand(2,2))
vL =update_vL(1, torch.rand(2,1), torch.rand(2,2), torch.rand(2,1), torch.rand(2,1), torch.rand(2,1), torch.rand(2,2))

print(vL)
print(vL.size())
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