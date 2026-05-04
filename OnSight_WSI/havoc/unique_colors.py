import random
from collections import OrderedDict


def reverse(colors):
    res = OrderedDict()
    for k, v in colors.items():
        res[k] = v[::-1]
    return res


def normalize(colors):
    res = OrderedDict()
    for k, v in colors.items():
        res[k] = tuple([x / 255. for x in v])
    return res


RGB_COLORS = OrderedDict({
    'red': (230, 25, 75), 'green': (60, 180, 75), 'yellow': (255, 225, 25),
    'blue': (0, 130, 200), 'orange': (245, 130, 48), 'purple': (145, 30, 180),
    'cyan': (70, 240, 240), 'magenta': (240, 50, 230), 'lime': (210, 245, 60),
    'grey': (128, 128, 128), 'brown': (170, 110, 40), 'black': (0, 0, 0),
    # not recommended to go more than 12 colors...
    'pink': (250, 190, 190), 'teal': (0, 128, 128), 'lavender': (230, 190, 255),
    'maroon': (128, 0, 0), 'mint': (170, 255, 195), 'olive': (128, 128, 0),
    'apricot': (255, 215, 180), 'navy': (0, 0, 128), 'beige': (255, 250, 200), 'white': (255, 255, 255)
})
BGR_COLORS = reverse(RGB_COLORS)
SCALED_RGB_COLORS = normalize(RGB_COLORS)
SCALED_BGR_COLORS = normalize(BGR_COLORS)


def next_color_generator(scaled=False, mode='rgb', shuffle=True, seed=10):
    '''
    Iterates over the possible color names and its value
    '''

    if mode == 'rgb':
        colors = RGB_COLORS
    else:
        colors = BGR_COLORS

    if scaled:
        colors = normalize(colors)

    if shuffle:
        colors = list(colors.items())
        random.Random(seed).shuffle(colors)
        colors = OrderedDict(colors)

    i = 0
    while True:
        i %= len(colors)
        color_name = list(colors)[i]
        yield dict(name=color_name, val=colors[color_name])
        i += 1
