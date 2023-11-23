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
Tf_map = {}
Texture_file = None
expose_v = -1
target_up = []
texcoord = []
bounces = 0
fisheye = 0
panorama = 0

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
    p1 = np.array(Vertices[ind1-1][0])
    p2 = np.array(Vertices[ind2-1][0])
    p3 = np.array(Vertices[ind3-1][0])
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
    texture_coords = list(np.multiply(b1, Vertices[ind1-1][1]) + np.multiply(b2,Vertices[ind2-1][1]) + np.multiply(b3,Vertices[ind3-1][1]))
    return [t, list(intersect_pt), normal, texture_coords]

def calc_color_sphere(cent, rsi, obj_color, direction, av_light, av_bulb_light, Texture_file):
    act_color = np.array([0, 0, 0])
    obj_color_array = np.array(obj_color)
    p = np.array(rsi)
    center = np.array(cent)
    normal = p - center
    normal = normal / np.linalg.norm(normal)
    if(Texture_file != None) :
        width, height = Tf_map[Texture_file]
        longitude = math.atan2(normal[0], normal[2])
        latitude = math.atan2(normal[1], math.sqrt(normal[0]**2 + normal[2]**2))

        pixel_x = int((longitude + math.pi) / (2 * math.pi) * width)
        pixel_y = int((-latitude + math.pi / 2) / math.pi * height)

        obj_color_array = np.array(Texture_file[pixel_x, pixel_y][:3]) / 255

    for light in av_light:
        light_dir_array = np.array(light[0])
        light_dir_array = light_dir_array/ np.sqrt(np.sum(light_dir_array**2))
        light_color_array = np.array(light[1])
        object_times_light = np.multiply(light_color_array, obj_color_array)
        light_dir = light_dir_array
        if(np.dot(normal, np.array(direction)) > 0):
            normal = -normal
        normal_dot_light = np.dot(normal, light_dir)
        if(normal_dot_light < 0):
            normal_dot_light = 0
        oln = np.multiply(object_times_light, normal_dot_light)
        act_color = np.add(act_color, oln)


    for light in av_bulb_light:
        light_dir_array = np.array(light[0]) - p
        light_dir_array = light_dir_array/ np.sqrt(np.sum(light_dir_array**2))
        light_color_array = np.array(light[1])
        object_times_light = np.multiply(light_color_array, obj_color_array)
        light_dir = light_dir_array
        if np.dot(normal, np.array(direction)) > 0:
            normal = -normal
        normal_dot_light = np.dot(normal, light_dir)
        if(normal_dot_light < 0):
            normal_dot_light = 0
        oln = np.multiply(object_times_light, normal_dot_light)
        oln = np.multiply(oln, 1 / magsq(p[0] - light[0][0], p[1] - light[0][1], p[2] - light[0][2]))
        act_color = np.add(act_color, oln)

    # exposure
    if(expose_v >= 0):
        act_color = 1 - np.exp(-expose_v * act_color)

    for i in range(3):
        if act_color[i] <= 0.0031308:
            act_color[i] *= 12.92
        else:
            act_color[i] = 1.055 * (act_color[i]**(1/2.4)) - 0.055
            act_color[i] = max(0, min(1, act_color[i]))

    return list(act_color)

def calc_color_plane(calc_normal, obj_color, direction, av_light, av_bulb_light, texcoord, Texture_file):
    act_color = np.array([0, 0, 0])
    obj_color_array = np.array(obj_color)
    if(texcoord != None):
        width, height = Tf_map[Texture_file]
        obj_color_array = np.array(Texture_file[int(texcoord[0]*width)-1, int(texcoord[1]*height)-1][:3]) / 255
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
        if act_color[i] <= 0.0031308:
            act_color[i] *= 12.92
        else:
            act_color[i] = 1.055 * (act_color[i]**(1/2.4)) - 0.055
            act_color[i] = max(0, min(1, act_color[i]))

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
            
            if fisheye == 1:
                if sx**2 + sy**2 >= 1:
                    continue        
                direction = [ray["forward"][0]*math.sqrt(1 - sx**2 - sy**2) + sx*ray["right"][0] + sy*ray["up"][0], 
                            ray["forward"][1]*math.sqrt(1 - sx**2 - sy**2) + sx*ray["right"][1] + sy*ray["up"][1], 
                            ray["forward"][2]*math.sqrt(1 - sx**2 - sy**2) + sx*ray["right"][2] + sy*ray["up"][2]]
            
            if panorama == 1:
                latitude = (y / Height) * math.pi - math.pi/2
                longitude = (x / Width) * 2 * math.pi - math.pi

                direction = [
                    math.cos(latitude) * math.sin(longitude),
                    -math.sin(latitude),
                    -math.cos(latitude) * math.cos(longitude)
                ]


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
                    intersection_data = [x, y, tri, direction, 2, rti[2], rti[3]]

            if intersection_point:
                intersect_to_xy[tuple(intersection_point)] = intersection_data
    
    shadows(intersect_to_xy)

def shadows(ixy):
    for points in ixy.keys():
        av_light = []
        av_bulb_light = []
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
        
        for light in Bulbs:
            flag_temp = 0
            ray["center"] = list(points)
            PtoL = np.array([points[0] - light[0][0], points[1] - light[0][1], points[2] - light[0][2]])
            dist = magsq(points[0] - light[0][0], points[1] - light[0][1], points[2] - light[0][2])
            for sphere in Spheres:
                if(sphere != ixy[points][2]):
                    rsi = ray_sphere_intersection(sphere[0], sphere[1], sphere[2], sphere[3], list(-PtoL))
                    if(rsi != "no intersection"):
                        PtoO = np.array([points[0] - rsi[1][0], points[1] - rsi[1][1], points[2] - rsi[1][2]])
                        dist2 = magsq(points[0] - rsi[1][0], points[1] - rsi[1][1], points[2] - rsi[1][2])
                        if(dist2 <= dist and np.dot(PtoO, PtoL) >= 0.0):
                            image.im.putpixel((ixy[points][0], ixy[points][1]), (0, 0, 0, 255))
                        else:
                            flag_temp += 1 
                    else:
                        flag_temp += 1
            for plane in Planes:
                if(plane != ixy[points][2]):
                    rpi = ray_plane_intersection(plane[0],plane[1],plane[2], plane[3], list(-PtoL))
                    if(rpi != "no intersection"):
                        dist2 = magsq(points[0] - rpi[1][0], points[1] - rpi[1][1], points[2] - rpi[1][2])
                        if(dist2 < dist):
                            image.im.putpixel((ixy[points][0], ixy[points][1]), (0, 0, 0, 255))
                        else:
                            flag_temp += 1
                    else:
                        flag_temp += 1
            for tri in Triangles:
                if(tri != ixy[points][2]):
                    rti = ray_triangle_intersection(tri[0],tri[1],tri[2], list(-PtoL))
                    if(rti != "no intersection"):
                        dist2 = magsq(points[0] - rti[1][0], points[1] - rti[1][1], points[2] - rti[1][2])
                        if(dist2 < dist):
                            image.im.putpixel((ixy[points][0], ixy[points][1]), (0, 0, 0, 255))
                        else:
                            flag_temp += 1
                    else:
                        flag_temp += 1
            if(flag_temp == len(Spheres) + len(Planes) + len(Triangles) - 1):
                av_bulb_light.append(light)
        # COLORING
        if ixy[points][4] == 0:
            color = calc_color_sphere([ixy[points][2][0], ixy[points][2][1], ixy[points][2][2]], points, ixy[points][2][4], ixy[points][3], av_light, av_bulb_light, ixy[points][2][5])
            image.im.putpixel((ixy[points][0], ixy[points][1]), (int(color[0]*255), int(color[1]*255), int(color[2]*255), 255))
        elif ixy[points][4] == 1:
            color = calc_color_plane(ixy[points][2][0:3], ixy[points][2][4], ixy[points][3], av_light, av_bulb_light, None, None)
            image.im.putpixel((ixy[points][0], ixy[points][1]), (int(color[0]*255), int(color[1]*255), int(color[2]*255), 255))
        elif ixy[points][4] == 2:
            color = calc_color_plane(ixy[points][5], ixy[points][2][3], ixy[points][3], av_light, av_bulb_light, ixy[points][6], ixy[points][2][4])
            image.im.putpixel((ixy[points][0], ixy[points][1]), (int(color[0]*255), int(color[1]*255), int(color[2]*255), 255))





if __name__ == "__main__":
    commands = ['png', 'sphere', 'sun', 'color', 'expose', 'eye', 'forward', 'up', 'plane', 'xyz', 'tri', 'texture', 'texcoord', 'fisheye', 'panorama', 'bulb']
    for line in file.readlines():
        words = line.split()
        if(len(words) == 0):
            continue
        if(words[0] == "png"):
            create_png(words[1], words[2], words[3])
        elif(words[0] == "sphere"):
            Spheres.append([float(words[1]), float(words[2]), float(words[3]), float(words[4]), curr_color, Texture_file])
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
            Vertices.append([[float(words[1]), float(words[2]), float(words[3])], texcoord])
        elif(words[0] == "tri"):
            Triangles.append([int(words[1]), int(words[2]), int(words[3]), curr_color, Texture_file])
        elif(words[0] == "texture"):
            if(words[1] != "none"):
                Tf = Image.open(words[1])
                tf_width, tf_height = Tf.size
                Texture_file = Tf.load()
                Tf_map[Texture_file] = (tf_width, tf_height)
            else:
                Texture_file = None
        elif(words[0] == "texcoord"):
            if(len(words) < 3):
                texcoord = [0, 0]
            texcoord = [float(words[1]), float(words[2])]
        elif(words[0] == "fisheye"):
            fisheye = 1
        elif(words[0] == "panorama"):
            panorama = 1
        elif(words[0] == "bulb"):
            Bulbs.append([[float(words[1]), float(words[2]), float(words[3])], curr_color])
    
    draw()    
    image.save("photos/" + imageName)
        
