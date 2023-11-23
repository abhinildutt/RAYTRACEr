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
Planes = []
Lights = []
Vertices = []
Triangles = []
Bulbs = []
expose_v = -1
target_up = []

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

def ray_plane_intersection(A, B, C, D, direction):
    mag = math.sqrt(magsq(direction[0], direction[1], direction[2]))
    direction[0] = direction[0] / mag
    direction[1] = direction[1] / mag
    direction[2] = direction[2] / mag

    normal = np.array([A, B, C])
    point  = - D / math.sqrt(A*A + B*B + C*C) * normal
    if(np.dot(direction,normal) == 0):
        return "no intersection"
    t = np.dot((point - ray["center"]), normal) / np.dot(direction,normal)
    if(t < 0):
        return "no intersection"
    intersect_pt = np.multiply(t, direction) + np.array(ray["center"])
    return [t, list(intersect_pt)]
    
def ray_triangle_intersection(ind1, ind2, ind3, direction):
    mag = math.sqrt(magsq(direction[0], direction[1], direction[2]))
    direction[0] = direction[0] / mag
    direction[1] = direction[1] / mag
    direction[2] = direction[2] / mag
    
    if(ind1 < 0):
        ind1 = len(Vertices) + ind1 + 1
    if(ind2 < 0):
        ind2 = len(Vertices) + ind2 + 1
    if(ind3 < 0):
        ind3 = len(Vertices) + ind3 + 1
    p1 = np.array(Vertices[ind1-1])
    p2 = np.array(Vertices[ind2-1])
    p3 = np.array(Vertices[ind3-1])
    normal = np.cross(p2 - p1, p3 - p1)
    point = p1
    if(np.dot(direction,normal) == 0):
        return "no intersection"
    t = np.dot((point - ray["center"]), normal) / np.dot(direction,normal)
    if(t <= 0):
        return "no intersection"
    intersect_pt = np.multiply(t, direction) + np.array(ray["center"])

    a1 = np.cross((p3 - p1), normal)
    a2 = np.cross((p2 - p1), normal)
    e1 = a1 / np.dot(a1, p2 - p1)
    e2 = a2 / np.dot(a2, p3 - p1)
    b2 = np.dot(e1, intersect_pt - p1)
    b3 = np.dot(e2, intersect_pt - p1)
    b1 = 1 - b2 - b3
    if b1 < 0 or b2 < 0 or b3 < 0:
        return "no intersection"
    return [t, list(intersect_pt), normal]

def calc_color_sphere(cent, rsi, obj_color, direction, av_light):
    act_color = np.array([0, 0, 0])
    obj_color_array = np.array(obj_color)
    for light in av_light:
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

    # exposure
    if(expose_v >= 0):
        act_color = 1 - np.exp(-expose_v * act_color)

    for i in range(3):
        if(act_color[i] < 0):
            act_color[i] = 0
        elif(act_color[i] > 1):
            act_color[i] = 1

    return list(act_color)

def calc_color_plane(calc_normal, obj_color, direction, av_light):
    act_color = np.array([0, 0, 0])
    obj_color_array = np.array(obj_color)
    for light in av_light:
        light_dir_array = np.array(light[0])
        light_dir_array = light_dir_array/ np.sqrt(np.sum(light_dir_array**2))
        light_color_array = np.array(light[1])
        object_times_light = np.multiply(light_color_array, obj_color_array)
        normal = np.array(calc_normal)
        light_dir = light_dir_array
        if(np.dot(normal, np.array(direction)) > 0):
            normal = -normal
        normal = normal/ np.sqrt(np.sum(normal**2))
        normal_dot_light = np.dot(normal, light_dir)
        oln = np.multiply(object_times_light, normal_dot_light)
        act_color = np.add(act_color, oln)

    # exposure
    if(expose_v >= 0):
        act_color = 1 - np.exp(-expose_v * act_color)

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
                    intersection_data = [x, y, sphere, direction, 0]
            
            for plane in Planes:
                rpi = ray_plane_intersection(plane[0], plane[1], plane[2], plane[3], direction)
                if(rpi != "no intersection" and rpi[0] < t_min):
                    t_min = rpi[0]
                    intersection_point = rpi[1]
                    intersection_data = [x, y, plane, direction, 1]

            for tri in Triangles:
                rti = ray_triangle_intersection(tri[0], tri[1], tri[2], direction)
                if(rti != "no intersection" and rti[0] < t_min):
                    t_min = rti[0]
                    intersection_point = rti[1]
                    intersection_data = [x, y, tri, direction, 2, rti[2]]

            if intersection_point:
                intersect_to_xy[tuple(intersection_point)] = intersection_data
 
    shadows(intersect_to_xy)

def shadows(ixy):
    for points in ixy.keys():
        av_light = []
        for light in Lights:
            flag_temp = 0
            ray["center"] = list(points)
            for sphere in Spheres:
                if(sphere != ixy[points][2]):
                    rsi = ray_sphere_intersection(sphere[0], sphere[1], sphere[2], sphere[3], light[0])
                    if(rsi != "no intersection"):
                        image.im.putpixel((ixy[points][0], ixy[points][1]), (0, 0, 0, 255)) 
                    else:
                        flag_temp += 1
            for plane in Planes:
                if(plane != ixy[points][2]):
                    rpi = ray_plane_intersection(plane[0],plane[1],plane[2], plane[3], light[0])
                    if(rpi != "no intersection"):
                        image.im.putpixel((ixy[points][0], ixy[points][1]), (0, 0, 0, 255)) 
                    else:
                        flag_temp += 1
            for tri in Triangles:
                if(tri != ixy[points][2]):
                    rti = ray_triangle_intersection(tri[0],tri[1],tri[2], light[0])
                    if(rti != "no intersection"):
                        image.im.putpixel((ixy[points][0], ixy[points][1]), (0, 0, 0, 255)) 
                    else:
                        flag_temp += 1
            if(flag_temp == len(Spheres) + len(Planes) + len(Triangles) - 1):
                av_light.append(light)
        
        # COLORING
        if ixy[points][4] == 0:
            color = calc_color_sphere([ixy[points][2][0], ixy[points][2][1], ixy[points][2][2]], points, ixy[points][2][4], ixy[points][3], av_light)
            image.im.putpixel((ixy[points][0], ixy[points][1]), (int(color[0]*255), int(color[1]*255), int(color[2]*255), 255))
        elif ixy[points][4] == 1:
            color = calc_color_plane(ixy[points][2][0:3], ixy[points][2][4], ixy[points][3], av_light)
            image.im.putpixel((ixy[points][0], ixy[points][1]), (int(color[0]*255), int(color[1]*255), int(color[2]*255), 255))
        elif ixy[points][4] == 2:
            color = calc_color_plane(ixy[points][5], ixy[points][2][3], ixy[points][3], av_light)
            image.im.putpixel((ixy[points][0], ixy[points][1]), (int(color[0]*255), int(color[1]*255), int(color[2]*255), 255))





if __name__ == "__main__":
    commands = ['png', 'sphere', 'sun', 'color', 'expose', 'eye', 'forward', 'up', 'plane', 'xyz', 'tri', 'bulb']
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
        elif(words[0] == "expose"):
            expose_v = float(words[1])
        elif(words[0] == "eye"):
            ray["center"] = [float(words[1]), float(words[2]), float(words[3])]
        elif(words[0] == "forward"):
            ray["forward"] = [float(words[1]), float(words[2]), float(words[3])]
            vec_right = np.cross(np.array(ray["forward"]), np.array(ray["up"]))
            ray["right"] = list(vec_right/np.linalg.norm(vec_right))
            vec_up = np.cross(np.array(ray["right"]), np.array(ray["forward"]))
            ray["up"] = list(vec_up/np.linalg.norm(vec_up))
            if np.dot(np.array(target_up), np.array(ray["up"])) < 0:
                ray["right"] = list(-vec_right/np.linalg.norm(vec_right))
                vec_up = np.cross(np.array(ray["right"]), np.array(ray["forward"]))
                ray["up"] = list(vec_up/np.linalg.norm(vec_up))
        elif(words[0] == "up"):
            target_up = [float(words[1]), float(words[2]), float(words[3])] 
        elif(words[0] == "plane"):
            Planes.append([float(words[1]), float(words[2]), float(words[3]), float(words[4]), curr_color])
        elif(words[0] == "xyz"):
            Vertices.append([float(words[1]), float(words[2]), float(words[3])])
        elif(words[0] == "tri"):
            Triangles.append([int(words[1]), int(words[2]), int(words[3]), curr_color])
        elif(words[0] == "bulb"):
            Bulbs.append([float(words[1]), float(words[2]), float(words[3]), curr_color])
    
    draw()    
    image.save("photos/" + imageName)
        
