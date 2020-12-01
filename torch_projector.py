# Utils
import torch
import torchvision.transforms as trans
from PIL import Image
import pathlib
from tqdm import tqdm

torch.cuda.set_device(0)

torch.pi = torch.acos(torch.zeros(1)).item() * 2

from matplotlib import cm

# "Credits : https://github.com/sunset1995/py360convert"

cmaps = [('Perceptually Uniform Sequential', [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
             'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
             'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
             'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
             'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
             'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
             'Pastel1', 'Pastel2', 'Paired', 'Accent',
             'Dark2', 'Set1', 'Set2', 'Set3',
             'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
             'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

getcolors = lambda n: cm.get_cmap('plasma', n).colors

#@profile
def getuvCenter(u_size=4, v_size=4):
    v = torch.linspace(90, -90, v_size)
    u = torch.linspace(-180, 180, u_size)
    vdiff = torch.abs(v[1] - v[0]).long()
    udiff = torch.abs(u[1] - u[0]).long()
    uvCenter = torch.stack(torch.meshgrid([v, u]), -1).reshape(-1, 2)
    return uvCenter, udiff, vdiff

#@profile
def Te2p(e_img, h_fov, v_fov, u_deg, v_deg, out_hw, in_rot_deg=torch.tensor([0.]), mode='bilinear'):
    '''
    e_img:   ndarray in shape of [H, W, *]
    h_fov,v_fov: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    b, c, h, w = e_img.shape

    h_fov, v_fov = h_fov * torch.pi / 180., v_fov * torch.pi / 180.

    in_rot = in_rot_deg * torch.pi / 180.

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * torch.pi / 180.
    v = v_deg * torch.pi / 180.
    xyz = Txyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = Txyz2uv(xyz)
    coor_xy = Tuv2coor(uv, torch.tensor([h], dtype=float), torch.tensor([w], dtype=float))
    mid = torch.tensor([w / 2., h / 2.]).reshape(1, 1, 2)
    cords = (coor_xy - mid) / mid
    pers_img = torch.nn.functional.grid_sample(input=e_img, grid=cords.unsqueeze(0).float(), align_corners=True,
                                               mode=mode)
    return pers_img, coor_xy

#@profile
def TgetCors(h_fov, v_fov, u_deg, v_deg, out_hw, in_rot_deg=torch.tensor([0.]), mode='bilinear'):
    '''
    e_img_shape:   [b,c,h,w]
    h_fov,v_fov: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''

    h_fov, v_fov = h_fov * torch.pi / 180., v_fov * torch.pi / 180.

    in_rot = in_rot_deg * torch.pi / 180.

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * torch.pi / 180.
    v = v_deg * torch.pi / 180.
    xyz = Txyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = Txyz2uv(xyz)
    coor_xy = Tuv2coor(uv, torch.tensor([h], dtype=float), torch.tensor([w], dtype=float))
    return coor_xy.long()

#@profile
def Tuv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = torch.split(uv, 1, -1)
    coor_x = (u / (2 * torch.pi) + 0.5) * w - 0.5
    coor_y = (-v / torch.pi + 0.5) * h - 0.5
    return torch.cat([coor_x, coor_y], -1)

#@profile
def Tcoor2uv(coorxy, h, w):
    coor_x, coor_y = torch.split(coorxy, 1, -1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * torch.pi
    v = -((coor_y + 0.5) / h - 0.5) * torch.pi
    return torch.cat([u, v], -1)

#@profile
def Tuv2unitxyz(uv):
    u, v = torch.split(uv, 1, -1)
    y = torch.sin(v)
    c = torch.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)

    return torch.cat([x, y, z], dim=-1)

#@profile
def Txyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = torch.split(xyz, 1, -1)
    u = torch.atan2(x, z)
    c = torch.sqrt(x ** 2 + z ** 2)
    v = torch.atan2(y, c)

    return torch.cat([u, v], -1)

#@profile
def Trotation_matrix(rad, ax):
    """
    rad : torch.tensor, Eg. torch.tensor([2.0])
    ax  : torch.tensor, Eg. [1,0,0] or [0,1,0] or [0,0,1]
    """
    ax = ax / torch.pow(ax, 2).sum()
    R = torch.diag(torch.cat([torch.cos(rad)] * 3))
    R = R + torch.outer(ax, ax) * (1.0 - torch.cos(rad))
    ax = ax * torch.sin(rad)
    R = R + torch.tensor([[0, -ax[2], ax[1]],
                          [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]], dtype=ax.dtype)
    return R

#@profile
def Txyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    out = torch.ones((*out_hw, 3), dtype=float)
    x_max = torch.tan(torch.tensor([h_fov / 2])).item()
    y_max = torch.tan(torch.tensor([v_fov / 2])).item()
    x_rng = torch.linspace(-x_max, x_max, out_hw[1], dtype=float)
    y_rng = torch.linspace(-y_max, y_max, out_hw[0], dtype=float)
    out[..., :2] = torch.stack(torch.meshgrid(x_rng, -y_rng), -1).permute(1, 0, 2)
    Rx = Trotation_matrix(v, torch.tensor([1, 0, 0], dtype=float))
    Ry = Trotation_matrix(u, torch.tensor([0, 1, 0], dtype=float))
    dots = (torch.tensor([[0, 0, 1]], dtype=float) @ Rx) @ Ry
    Ri = Trotation_matrix(in_rot, dots[0])
    return ((out @ Rx) @ Ry) @ Ri

#@profile
def applyProjection(img_dim = (1,3,320,640),
                    gridu=4,
                    gridv=4,
                    pad=4,
                    sample_factor=10,
                    early_interpolation = False):
    uvCenter, udiff, vdiff = getuvCenter(gridu, gridv)
    colors = [torch.tensor(c[:-1]).reshape(1,-1,1,1) for c in getcolors(gridu * gridv).tolist()]
    h_fov = torch.tensor([udiff + pad * 2])
    v_fov = torch.tensor([vdiff + pad])
    scale_hw_max = (v_fov.item() * sample_factor, h_fov.item() * sample_factor)
    scale_hw_default  = (v_fov.item(), h_fov.item())
    if early_interpolation:
        scale_hw = scale_hw_max
    else:
        scale_hw = scale_hw_default
    fovlist = []
    for i, (v, u) in tqdm(enumerate(uvCenter)):
        _,idx = Te2p(e_img = torch.rand(*img_dim),
                h_fov = h_fov,
                v_fov = v_fov,
                u_deg = torch.tensor([u]),
                v_deg = torch.tensor([v]),
                out_hw = scale_hw,
                in_rot_deg = torch.tensor([180.]),
                mode = 'nearest')
        fovlist.append(idx)

    #@profile
    def morphNNFunc(nn_func=None, e_img=None):
        nn_func = nn_func if nn_func else lambda x,i:colors[i].cuda()
        for i, idx in enumerate(fovlist):
            if not early_interpolation:
                idx = idx.permute(2, 0, 1)
                idx = torch.nn.functional.interpolate(idx.unsqueeze(0), size = scale_hw_max, mode = "bilinear", align_corners=True)[0]
                idx = idx.permute(1,2,0)
            idx_chunked = idx.long().chunk(2, 2)
            e_img[:, :, idx_chunked[1].squeeze(), idx_chunked[0].squeeze()] = nn_func(e_img[:, :, idx_chunked[1].squeeze(), idx_chunked[0].squeeze()],i)
        return e_img
    return morphNNFunc


if __name__ == '__main__':
    import time
    n = 1
    then = time.time()
    g = 7
    morpher = applyProjection(gridu=g, gridv=g, pad=4, sample_factor = 50, early_interpolation=True)
    imgs = [*pathlib.Path("/home/keshav/data/finalEgok360/images/Desk_work/Desk_work/0547/").glob('*')][:n]
    imgs = torch.stack([*map(lambda x: trans.ToTensor()(Image.open(x)), imgs)]).cuda()
    now = time.time()
    print(now-then)
    out = morpher(nn_func=None, e_img=imgs)
    #trans.ToPILImage()(out[0]).save("torchout/torch_late_case_7_50_25.png")