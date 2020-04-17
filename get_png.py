#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019 Tamflex
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# reference:
# http://stackoverflow.com/questions/4458696/finding-out-what-characters-a-font-supports

import os
import re
import sys
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from itertools import chain
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode


def load_font(path):
    ttf = TTFont(
        path, 0, verbose=0, allowVID=0,
        ignoreDecompileErrors=True, fontNumber=-1)
    chars = chain.from_iterable(
        [y + (Unicode[y[0]], )
        for y in x.cmap.items()] for x in ttf['cmap'].tables)
    chars = set(chars)
    chars = sorted(chars, key=lambda c: int(c[0]))

    # Use this for just checking if the font contains the codepoint given as
    # second argument:
    # char = int(sys.argv[2], 0)
    # print(Unicode[char])
    # print(char in (x[0] for x in chars))

    ttf.close()
    return chars


def convert_ttf(path, fontsize=60, w=300, h=300):
    base = os.path.basename(path)
    dirs = os.path.splitext(base)[0]
    if not os.path.exists(dirs):
        os.mkdir(dirs)
    chars = load_font(path)
    font = ImageFont.truetype(path, fontsize)
    for c in chars:
        if re.match('.notdef|nonmarkingreturn|.null', c[1]):
            continue
        img = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        (ws, hs) = font.getsize(chr(c[0]))
        wb = (w - ws) * 0.5
        hb = (h - hs) * 0.5
        draw.text((wb, hb), chr(c[0]), (0, 0, 0), font=font)
        draw = ImageDraw.Draw(img)
        img.save('{}/{:d}.png'.format(dirs, c[0]))


if __name__ == '__main__':

    # chars = load_font(sys.argv[1])
    # for c in chars:
        # print(c[0],c[1],c[2])
    pat = 'testline.ttf'
    convert_ttf(pat, 120)
    # convert_ttf(sys.argv[1], 80)