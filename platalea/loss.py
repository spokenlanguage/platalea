import torch

def contrastive(M, margin=0.2):       
    "Returns contrastive margin loss over similarity matrix M."     
    E = - M
    D = torch.diag(E)
    C_c = torch.clamp(margin - E + D, min=0)
    C_r = torch.clamp(margin - E + D.view(-1,1), min=0)
    C = C_c + C_r
    return (C.sum() - torch.diag(C).sum())/C.size(0)**2

def cosine_matrix(U, V):
    "Returns the matrix of cosine similarity between each row of U and each row of V."
    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return torch.matmul(U_norm, V_norm.t())
