from PIL import Image
import math
import numpy as np
from sys import argv

#file related data
file = open(argv[1], "r")
command = ""
imageName = ""
image = None
Width = 0
Height = 0
attribute = []

ray = {
    "center" : [0,0,0],
    "forward" : [0,0,-1],
    "right" : [1,0,0],
    "up" : [0,1,0],
}

def create_png(width, height, name):
    global image
    global imageName
    global Width
    global Height
    Width = int(width)
    Height = int(height)
    image = Image.new("RGBA", (Width, Height), (0, 0, 0, 1))
    imageName = name

def distsq(x, y, z, xx, yy, zz):
    return (xx - x)**2 + (yy - y)**2 + (zz - z)**2

def magsq(x, y, z) {
    return x**2 + y**2 + z**2
}

def ray_sphere_intersection(x, y, z, radius):
    inside = 1 if (distsq(x, y, z, ray["center"][0], ray["center"][1], ray["center"][2])) < radius**2 else 0
    tc = (x - ray["center"][0])*ray["forward"][0] + (y - ray["center"][1])*ray["forward"][1] + (z - ray["center"][2])*ray["forward"][2]
    if (not inside and tc < 0):
        return "no intersection"
    d2 = magsq(ray["center"][0] + tc[0]*ray["forward"][0] - x, ray["center"][1] + tc[1]*ray["forward"][1] - y, ray["center"][2] + tc[2]*ray["forward"][2] - z) 
    if (not inside and radius**2 < d2):
        return "no intersection"
    t_off = math.sqrt(radius**2  - d2)
    if(inside):
        t = tc + t_off
        return [t, [t*ray["forward"][0] + ray["center"][0], t*ray["forward"][1] + ray["center"][1], t*ray["forward"][2] + ray["center"][2]]]
    t = tc - t_off
    return [t, [t*ray["forward"][0] + ray["center"][0], t*ray["forward"][1] + ray["center"][1], t*ray["forward"][2] + ray["center"][2]]]

if __name__ == "__main__":
    commands = ['png']
    for line in file.readlines():
        words = line.split()
        if(len(words) == 0):
            break
        if(words[0] == "png"):
            create_png(words[1], words[2], words[3])
    image.save("photos/" + imageName)
        
