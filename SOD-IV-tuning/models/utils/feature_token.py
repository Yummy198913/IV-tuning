def token2feature(tokens, H, W):
    B, L, D = tokens.shape
    x = tokens.permute(0, 2, 1).view(B, D, H, W).contiguous()
    return x


def feature2token(x):
    B, C, H, W = x.shape
    L = H * W
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens