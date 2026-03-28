import torch
import torch.nn.functional as F
import math, random

def morphomix_simple(images, targets, num_classes, preset='standard', use_inter=True, onehot=True):
    """
    MorphoMix (简化版)：结构保真增强（类内Mix + 可选类间对偶扰动）
    - images: FloatTensor [N,C,H,W] in [0,1]
    - targets: LongTensor [N] 或 one-hot [N,num_classes]
    - num_classes: 类别数
    - preset: 'light' | 'standard' | 'strong'
    - use_inter: 是否启用类间对偶扰动（默认 True）
    - onehot: 是否返回 one-hot/soft 标签（默认 True）
    """
    assert images.dtype.is_floating_point
    device = images.device
    N, C, H, W = images.shape

    # 预设超参（全部内置）
    cfgs = {
        'light':   dict(p_intra=0.2, p_inter=0.1, lam_alpha=0.5, morph_prob=0.5, blur_p=0.2, cut_p=0.2, edge_crop=1),
        'standard':dict(p_intra=0.3, p_inter=0.2, lam_alpha=0.4, morph_prob=0.7, blur_p=0.3, cut_p=0.3, edge_crop=1),
        'strong':  dict(p_intra=0.4, p_inter=0.3, lam_alpha=0.3, morph_prob=0.8, blur_p=0.4, cut_p=0.4, edge_crop=2),
    }
    cfg = cfgs.get(preset, cfgs['standard'])
    if not use_inter:
        cfg['p_inter'] = 0.0

    # 标签处理
    if targets.dim() == 2:
        y = targets.argmax(1)
        onehot_in = True
    else:
        y = targets
        onehot_in = False

    imgs = images.clone()

    # —— 边缘轻裁（模拟残片缺边）
    if cfg['edge_crop'] > 0:
        pad = cfg['edge_crop']
        imgs = imgs[:, :, pad:H-pad, pad:W-pad]
        imgs = F.pad(imgs, (pad,pad,pad,pad), mode='replicate')

    # —— 类内 Mix（同类配对）
    n_intra = int(cfg['p_intra'] * N)
    if n_intra > 0:
        idx = torch.randperm(N, device=device)[:n_intra]
        partners = []
        for i in idx.tolist():
            same = (y == y[i]).nonzero(as_tuple=False).flatten()
            j = i if len(same)==0 else same[torch.randint(0, len(same), (1,)).item()]
            partners.append(j)
        partners = torch.tensor(partners, device=device)
        lam = torch.distributions.Beta(cfg['lam_alpha'], cfg['lam_alpha']).sample((n_intra,)).to(device).view(-1,1,1,1)
        imgs[idx] = lam * imgs[idx] + (1-lam) * imgs[partners]

    # —— 类间对偶扰动 + 软标签（可关）
    n_inter = int(cfg['p_inter'] * N)
    target_out = None
    if n_inter > 0:
        idx = torch.randperm(N, device=device)[:n_inter]
        partners = []
        for i in idx.tolist():
            # 简单随机找异类（最多重试5次）
            j = torch.randint(0, N, (1,), device=device).item()
            if y[j] == y[i]:
                for _ in range(5):
                    j = torch.randint(0, N, (1,), device=device).item()
                    if y[j] != y[i]: break
            partners.append(j)
        partners = torch.tensor(partners, device=device)

        xa = imgs[idx].clone()
        xb = imgs[partners].clone()

        # 形态扰动（erode/dilate + blur + thin cutout），强度内置固定
        def erode(x, k=1):
            xpad = F.pad(x, (k,k,k,k), mode='replicate')
            return -F.max_pool2d(-xpad, kernel_size=2*k+1, stride=1)
        def dilate(x, k=1):
            xpad = F.pad(x, (k,k,k,k), mode='replicate')
            return F.max_pool2d(xpad, kernel_size=2*k+1, stride=1)
        def gblur(x, k=3, s=0.8):
            dev = x.device
            ax = torch.arange(k, device=dev) - (k-1)/2
            k1 = torch.exp(-0.5*(ax/s)**2); k1 /= k1.sum()
            kx = k1.view(1,1,1,k); ky = k1.view(1,1,k,1)
            C = x.size(1)
            x = F.conv2d(x, kx.expand(C,1,1,k), padding=(0,k//2), groups=C)
            x = F.conv2d(x, ky.expand(C,1,k,1), padding=(k//2,0), groups=C)
            return x
        def thin_cutout(x, num=2):
            B, C, Hh, Ww = x.shape
            for b in range(B):
                m = torch.ones((Hh, Ww), dtype=torch.bool, device=x.device)
                for _ in range(num):
                    length = random.randint(Hh//8, Hh//4)
                    width  = random.randint(1, 3)
                    y0 = random.randint(0, Hh-1); x0 = random.randint(0, Ww-1)
                    th = random.uniform(0, 2*math.pi)
                    dy, dx = math.sin(th), math.cos(th)
                    for t in range(length):
                        yy = int(y0 + dy*t); xx = int(x0 + dx*t)
                        if 0<=yy<Hh and 0<=xx<Ww:
                            yy0 = max(0, yy-width//2); yy1 = min(Hh, yy+(width+1)//2)
                            xx0 = max(0, xx-width//2); xx1 = min(Ww, xx+(width+1)//2)
                            m[yy0:yy1, xx0:xx1] = False
                x[b] = x[b] * m
            return x

        # 对 a：随机 erode/dilate, blur, cutout（按内置概率）
        for t in range(xa.size(0)):
            if random.random() < cfg['morph_prob']:
                if random.random() < 0.5: xa[t:t+1] = erode(xa[t:t+1], k=1)
                else:                      xa[t:t+1] = dilate(xa[t:t+1], k=1)
            if random.random() < cfg['blur_p']:
                xa[t:t+1] = gblur(xa[t:t+1], k=3, s=0.8)
            if random.random() < cfg['cut_p']:
                xa[t:t+1] = thin_cutout(xa[t:t+1], num=2)

        # 对 b：轻仿射（内置固定范围）
        def affine(x):
            B, C, Hh, Ww = x.shape
            thetas=[]
            for _ in range(B):
                ang = random.uniform(-5, 5) * math.pi/180
                s   = random.uniform(0.97, 1.03)
                tx  = random.uniform(-3, 3)*2.0/Ww
                ty  = random.uniform(-3, 3)*2.0/Hh
                a = s*math.cos(ang); b = -s*math.sin(ang)
                c = s*math.sin(ang); d =  s*math.cos(ang)
                thetas.append(torch.tensor([[a,b,tx],[c,d,ty]], device=x.device))
            theta = torch.stack(thetas,0)
            grid  = F.affine_grid(theta, size=x.size(), align_corners=False)
            return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        xb = affine(xb)

        lam = torch.distributions.Beta(cfg['lam_alpha'], cfg['lam_alpha']).sample((n_inter,)).to(device).view(-1,1,1,1)
        imgs[idx] = lam*xa + (1-lam)*xb

        # 软标签（建议）
        ya = F.one_hot(y[idx],       num_classes=num_classes).float()
        yb = F.one_hot(y[partners],  num_classes=num_classes).float()
        lam_vec = lam[:,0,0,0].unsqueeze(1)
        soft = lam_vec*ya + (1-lam_vec)*yb
        target_out = F.one_hot(y, num_classes=num_classes).float()
        target_out[idx] = soft

    # —— 输出标签
    if target_out is not None:
        return imgs.clamp(0,1), target_out
    if targets.dim()==2:
        return imgs.clamp(0,1), targets
    if onehot:
        return imgs.clamp(0,1), F.one_hot(y, num_classes=num_classes).float()
    return imgs.clamp(0,1), y


def collate_with_morphomix(batch, num_classes, preset='standard', use_inter=True, onehot=True):
    # batch: list of (image, label)；image [C,H,W]，label int
    imgs = torch.stack([b[0] for b in batch], dim=0)      # [N,C,H,W]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)  # [N]
    imgs = imgs.float()
    # 如果你的图像是 [0,255]，这里要 /255
    if imgs.max() > 1.0: imgs = imgs / 255.0

    imgs_aug, targets_aug = morphomix_simple(
        imgs, labels, num_classes=num_classes,
        preset=preset, use_inter=use_inter, onehot=onehot
    )
    return imgs_aug, targets_aug