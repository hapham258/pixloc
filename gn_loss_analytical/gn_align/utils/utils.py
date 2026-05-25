import torch

def inverse2x2(A, valid, thresh = 5):
    B = torch.empty_like(A)
    a = A[...,0,0]
    b = A[...,0,1]
    c = A[...,1,0]
    d = A[...,1,1]

    det = B[...,1,1];
    det = (a*d - b*c)
    if valid is None:
        valid = (det > thresh) 
    else:
        valid = valid & (det > thresh) 

    B[...,0,0] = d/det
    B[...,0,1] = -b/det
    B[...,1,0] = -c/det
    B[...,1,1] = a/det
    return B, valid

