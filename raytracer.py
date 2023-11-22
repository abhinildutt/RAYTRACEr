from PIL import Image
import math
import numpy as np
from sys import argv
from collections import defaultdict

#file related data
file = open(argv[1], "r")
imageName = ""
image = None
curr_color = [1, 1, 1]
Width = 0
Height = 0

ray = {
    "center" : [0,0,0],
    "forward" : [0,0,-1],
    "right" : [1,0,0],
    "up" : [0,1,0],
}

Spheres = []
Lights = []

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

def magsq(x, y, z):
    return x**2 + y**2 + z**2

def ray_sphere_intersection(x, y, z, radius, direction):
    mag = math.sqrt(magsq(direction[0], direction[1], direction[2]))
    direction[0] = direction[0] / mag
    direction[1] = direction[1] / mag
    direction[2] = direction[2] / mag

    inside = 1 if (distsq(x, y, z, ray["center"][0], ray["center"][1], ray["center"][2])) < radius**2 else 0
    tc = (x - ray["center"][0])*direction[0] + (y - ray["center"][1])*direction[1] + (z - ray["center"][2])*direction[2]
    if (not inside and tc < 0):
        return "no intersection"
    d2 = magsq(ray["center"][0] + tc*direction[0] - x, ray["center"][1] + tc*direction[1] - y, ray["center"][2] + tc*direction[2] - z) 
    if (not inside and radius**2 < d2):
        return "no intersection"
    t_off = math.sqrt(radius**2  - d2)
    if(inside):
        t = tc + t_off
        return [t, [t*direction[0] + ray["center"][0], t*direction[1] + ray["center"][1], t*direction[2] + ray["center"][2]]]
    t = tc - t_off
    return [t, [t*direction[0] + ray["center"][0], t*direction[1] + ray["center"][1], t*direction[2] + ray["center"][2]]]

def calc_color(x, y, cent, rsi, obj_color, direction):
    act_color = np.array([0, 0, 0])
    obj_color_array = np.array(obj_color)
    for light in Lights:
        light_dir_array = np.array(light[0])
        light_dir_array = light_dir_array/ np.sqrt(np.sum(light_dir_array**2))
        light_color_array = np.array(light[1])
        object_times_light = np.multiply(light_color_array, obj_color_array)
        p = np.array(rsi)
        center = np.array(cent)
        normal = p - center
        light_dir = light_dir_array
        if(np.dot(normal, np.array(direction)) > 0):
            normal = -normal
        normal = normal/ np.sqrt(np.sum(normal**2))
        normal_dot_light = np.dot(normal, light_dir)
        oln = np.multiply(object_times_light, normal_dot_light)
        act_color = np.add(act_color, oln)

    for i in range(3):
        if(act_color[i] < 0):
            act_color[i] = 0
        elif(act_color[i] > 1):
            act_color[i] = 1

    return list(act_color)
    


def draw():
    intersect_to_xy = defaultdict(None)
    for x in range(Width):
        for y in range(Height):
            sx = (2*x - Width)/ max(Width, Height)
            sy = (Height - 2*y)/ max(Width, Height)
            direction = [ray["forward"][0] + sx*ray["right"][0] + sy*ray["up"][0], 
                         ray["forward"][1] + sx*ray["right"][1] + sy*ray["up"][1], 
                         ray["forward"][2] + sx*ray["right"][2] + sy*ray["up"][2]]
            
            t_min = math.inf
            intersection_point = None
            intersection_data = None

            for sphere in Spheres:
                rsi = ray_sphere_intersection(sphere[0], sphere[1], sphere[2], sphere[3], direction)        
                if(rsi != "no intersection" and rsi[0] < t_min):
                    t_min = rsi[0]
                    intersection_point = rsi[1]
                    intersection_data = [x, y, sphere, direction]
            
            if intersection_point:
                intersect_to_xy[tuple(intersection_point)] = intersection_data
 

    for points in intersect_to_xy.keys():
        ixy = intersect_to_xy[points]
        color = calc_color(ixy[0], ixy[1], [ixy[2][0], ixy[2][1], ixy[2][2]], points, ixy[2][4], ixy[3])
        image.im.putpixel((ixy[0], ixy[1]), (int(color[0]*255), int(color[1]*255), int(color[2]*255), 255))

    shadows(intersect_to_xy)

def shadows(ixy):
    for points in ixy.keys():
        for light in Lights:
            ray["center"] = list(points)
            for sphere in Spheres:
                if(sphere != ixy[points][2]):
                    rsi = ray_sphere_intersection(sphere[0], sphere[1], sphere[2], sphere[3], light[0])
                    if(rsi != "no intersection"):
                        image.im.putpixel((ixy[points][0], ixy[points][1]), (0, 0, 0, 255))
    ray["center"] = [0, 0, 0]

        


if __name__ == "__main__":
    commands = ['png', 'sphere', 'sun', 'color']
    for line in file.readlines():
        words = line.split()
        if(len(words) == 0):
            continue
        if(words[0] == "png"):
            create_png(words[1], words[2], words[3])
        elif(words[0] == "sphere"):
            Spheres.append([float(words[1]), float(words[2]), float(words[3]), float(words[4]), curr_color])
        elif(words[0] == "sun"):
            Lights.append([[float(words[1]), float(words[2]), float(words[3])], curr_color])
        elif(words[0] == "color"):
            curr_color = [float(elem) for elem in words[1:]]
    draw()    
    image.save("photos/" + imageName)
        
