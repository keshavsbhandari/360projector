from scipy.ndimage import map_coordinates
import numpy as np
from matplotlib import cm
from tqdm import tqdm
import pathlib
from PIL import Image

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
def rotation_matrix(rad, ax):
    ax = np.array(ax)
    assert len(ax.shape) == 1 and ax.shape[0] == 3
    ax = ax / np.sqrt((ax ** 2).sum())
    R = np.diag([np.cos(rad)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))
    ax = ax * np.sin(rad)
    R = R + np.array([[0, -ax[2], ax[1]],
                      [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])

    return R

#@profile
def xyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    out = np.ones((*out_hw, 3), np.float32)
    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = rotation_matrix(v, [1, 0, 0])
    Ry = rotation_matrix(u, [0, 1, 0])
    dots = np.array([0, 0, 1.0]).dot(Rx).dot(Ry)
    Ri = rotation_matrix(in_rot, dots)
    return out.dot(Rx).dot(Ry).dot(Ri)

#@profile
def xyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x ** 2 + z ** 2)
    v = np.arctan2(y, c)

    return np.concatenate([u, v], axis=-1)

#@profile
def uv2unitxyz(uv):
    u, v = np.split(uv, 2, axis=-1)
    y = np.sin(v)
    c = np.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)

    return np.concatenate([x, y, z], axis=-1)

#@profile
def uv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return np.concatenate([coor_x, coor_y], axis=-1)

#@profile
def coor2uv(coorxy, h, w):
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi
    v = -((coor_y + 0.5) / h - 0.5) * np.pi

    return np.concatenate([u, v], axis=-1)

#@profile
def sample_equirec(e_img, coor_xy, order):
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0)
    return map_coordinates(e_img, [coor_y, coor_x],
                           order=order, mode='wrap')[..., 0]

#@profile
def e2p(e_img, h_fov, v_fov, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear'):
    '''
    e_img:   ndarray in shape of [H, W, *]
    h_fov,v_fov: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]

    try:
        h_fov, v_fov = h_fov * np.pi / 180, v_fov * np.pi / 180
    except:
        h_fov, v_fov = fov, fov
    in_rot = in_rot_deg * np.pi / 180

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180
    xyz = xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    pers_img = np.stack([
        sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    return pers_img, coor_xy[...,::-1]

#@profile
def get_uv_center(u_size=4, v_size=4):
    v = np.linspace(90, -90, u_size)
    u = np.linspace(-180, 180, v_size)
    vdiff = int(abs(v[0] - v[1]))
    udiff = int(abs(u[0] - u[1]))
    mesh = np.stack(np.meshgrid(v, u), -1).reshape(-1, 2)
    return mesh, udiff, vdiff

#@profile
def applyProjection(img_dim = (320,640,3),
                    gridu=4,
                    gridv=4,
                    pad=4,
                    sample_factor=10):
    uvCenter, udiff, vdiff = get_uv_center(gridu, gridv)
    colors = [c[:-1] for c in getcolors(gridu * gridv).tolist()]
    h_fov = udiff + pad * 2
    v_fov = vdiff + pad
    scale_hw = (v_fov * sample_factor, h_fov * sample_factor)
    fovlist = []
    for i, (v, u) in tqdm(enumerate(uvCenter)):
        _, idx = e2p(e_img=np.random.rand(*img_dim).astype(np.uint8),
                     h_fov=h_fov,
                     v_fov=v_fov,
                     u_deg=u,
                     v_deg=v,
                     out_hw=scale_hw,
                     in_rot_deg=180,
                     mode='bilinear')
        fovlist.append(idx)

    #@profile
    def morphNNFunc(nn_func=None, e_imgs=None):
        nn_func = nn_func if nn_func else lambda x,i:np.array(colors[i]).reshape(1,3)*255
        for j,e_img in tqdm(enumerate(e_imgs)):
            for i, idx in enumerate(fovlist):
                mask = tuple(idx.astype(np.long).reshape(-1,2).T)
                e_img[mask] = nn_func(e_img[mask],i)
            e_imgs[j] = e_img
        return e_imgs
    return morphNNFunc

if __name__ == '__main__':
    import time
    then = time.time()
    n = 1
    morpher = applyProjection(gridu=7,
                    gridv=7,
                    pad=4,
                    sample_factor=10)
    imgs = [*pathlib.Path("/home/keshav/data/finalEgok360/images/Desk_work/Desk_work/0547/").glob('*')][:n]
    print("NUMBER OF IMAGES",len(imgs))
    imgs = np.stack([*map(lambda x:np.array(Image.open(x)), imgs)])
    now = time.time()
    print(now - then)
    out = morpher(nn_func=None, e_imgs=imgs)