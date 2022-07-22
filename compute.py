import os
import math
import pymesh
from SSPPIopts import SSPPIopts
import sys
import time
import numpy as np
from sklearn.neighbors import KDTree
kdtime = 0
def producemesh(vert):
    start = time.time()
    vertice = []
    p1 = np.array(vert[-1])
    vert = np.delete(vert,-1,axis = 0)
    kdt = KDTree(np.array(vert))
    d,r = kdt.query([p1])
    vertice.append(p1)
    vertice.append(vert[int(r)])
    while(len(vert)>1 ):
        queryp = vert[int(r)]
        vert = np.delete(vert,r,axis = 0)
        kdt = KDTree(np.array(vert))
        d,r = kdt.query([queryp])
        vertice.append(vert[int(r)])
    end = time.time()
    global kdtime
    kdtime = kdtime + end - start
    return vertice
def computearea(vert,num):
    inter_vert = []
    tmp = vert.split('\t')
    p1 = tmp[1].split(' ')
    ori_p1 = []
    n1 = tmp[0].split(' ')
    p1[0] = float(p1[0]) + float(num)*float(n1[0])
    p1[1] = float(p1[1]) + float(num)*float(n1[1])
    p1[2] = float(p1[2]) + float(num)*float(n1[2])
    p2 = []
    p3 = []
    n2 = []
    for i in range(len(tmp)-3):
        npp = tmp[i+3].split(' ')
        n2.append(npp[0:3])
        p2.append(npp[3:6])
        p3.append(npp[6:9])
    
    for i in range(len(p2)):
        if(float(n2[i][0])*float(n1[0])+float(n2[i][1])*float(n1[1])+float(n2[i][2])*float(n1[2]) == float(0)):
            continue
        else:
            m = ((float(p1[0]) - float(p2[i][0])) * float(n1[0]) +\
                (float(p1[1]) - float(p2[i][1])) * float(n1[1]) +\
                (float(p1[2]) - float(p2[i][2])) * float(n1[2])) / (float(n1[0]) * float(n2[i][0]) + float(n1[1]) * float(n2[i][1]) + float(n1[2]) * float(n2[i][2]))
            vert_tmp = [float(p2[i][0])+float(n2[i][0])*m, float(p2[i][1])+float(n2[i][1])*m,float(p2[i][2])+float(n2[i][2])*m]
            inter_vert.append(vert_tmp)
    inter_vert.append(p1)
   # inter_vert = producemesh(inter_vert)
    face = []
    for i in range(len(inter_vert)-2):
        face.append([0,i+1,i+2])

    face.append([0,1,len(inter_vert)-1])
            
    mesh = pymesh.form_mesh(np.array(inter_vert),np.array(face))
    mesh.add_attribute("face_area")
    subarea = mesh.get_attribute("face_area")
    sumarea = 0
    for i in subarea:
        sumarea = sumarea + i
    return sumarea,inter_vert,face

def revert(str_1):
    p = str_1.split('\t')
    ptmp = p[0].split(' ')
    vert = []
    idx = 3
    dict = {}
    for i in range(len(p)-3):
        tmp = p[i + 3].split(' ')
        for j in range(len(tmp)-8):
            items = {tmp[j+3]+tmp[j+4]+tmp[j+5]:idx}
            dict.update(items)
            vert.append([float(tmp[j+3]),float(tmp[j+4]),float(tmp[j+5])])
            idx = idx + 1

    vert.append([float(ptmp[0]),float(ptmp[1]),float(ptmp[2])])


    vert_mesh = producemesh(vert)

    str_later = ''
    for i in range(3):
        str_later = str_later + p[i] + '\t'
    for i in range(len(vert_mesh)-1):
        ori = dict.get(str(vert_mesh[i+1][0])+str(vert_mesh[i+1][1])+str(vert_mesh[i+1][2]))
        str_later = str_later + p[ori] + '\t'
    return str_later.rstrip('\t')

def compute(vert,index):
    vert = revert(vert)
    vert_sum = []
    face_sum = []
    area,vert_,face_ = computearea(vert,0)
    area_1,vert1,face1 = computearea(vert,-0.1)
    area_2,vert2,face2 = computearea(vert,0.1)
    if area_1 < area:
        minarea = area_1
        distance = -0.1
        vert_sum.append(vert_)
        vert_sum.append(vert1)
        face_sum.append(face_)
        face_sum.append(face1)
        for i in range(25):
            area_1,vert1,face1 = computearea(vert,-0.1-float((i+1))/10)
            if area_1 - minarea < 0:
                minarea = area_1
                distance = distance -0.1
                vert_sum.append(vert1)
                face_sum.append(face1)
            else:
                break
    else:
        minarea = area_2
        distance = 0.1
        vert_sum.append(vert_)
        vert_sum.append(vert2)
        face_sum.append(face_)
        face_sum.append(face2)

        for i in range(25):
            area_2,vert2,face2 = computearea(vert,0.1+float((i+1))/10)
            if area_2 - minarea < 0 :
                minarea = area_2
                distance = distance + 0.1
                vert_sum.append(vert2)
                face_sum.append(face2)
            else:
                break
    vertarray = np.array(vert_sum)
    vertarray_new = np.reshape(vertarray,[-1,3])
    for i in range(len(face_sum)):
        for j in range(len(face_sum[i])):
            for k in range(len(face_sum[i][j])):
                face_sum[i][j][k] = face_sum[i][j][k]+i*len(vert_sum[0])
    facearray = np.array(face_sum)
    facearray_new = np.reshape(facearray,[-1,3])
    mesh = pymesh.form_mesh(vertarray_new,facearray_new)
    tmp = vert.split('\t')
    p1 = tmp[1].split(' ')
    n1 = tmp[0].split(' ')
    realdistance = math.sqrt(math.pow(float(distance)*float(n1[0]),2)+math.pow(float(distance)*float(n1[1]),2)+math.pow(float(distance)*float(n1[2]),2))
    if distance < 0:
        realdistance = 0 - realdistance
    p1x = float(p1[0]) + float(distance)*float(n1[0])
    p1y = float(p1[1]) + float(distance)*float(n1[1])
    p1z = float(p1[2]) + float(distance)*float(n1[2])
    p2 = []
    n2 = []
    for i in range(len(tmp)-3):
        npp = tmp[i+3].split(' ')
        n2.append(npp[0:3])
        p2.append(npp[3:6])
    
    #costheata = []

    #for i in range(len(p2)):
     #   cos_ = abs((abs(float(n2[i][0])*p1x+float(n2[i][1])*p1y+float(n2[i][2])*p1z-(float(n2[i][0])*float(p2[i][0])+float(n2[i][1])*float(p2[i][1])+float(n2[i][2])*float(p2[i][2]))) / math.sqrt(math.pow(float(n2[i][0]),2)+math.pow(float(n2[i][1]),2)+math.pow(float(n2[i][2]),2))) / realdistance)
      #  costheata.append(cos_)

    #npcos = np.array(costheata)
    #cos_min = np.min(npcos)
    #cos_max = np.max(npcos)
    #cos_mean = np.mean(npcos)
    areaaboutdistance = (area-minarea)/(realdistance)
    end = time.time()
    #pymesh.save_mesh("ply/test"+str(index)+'_'+str(float('%.2f'%realdistance))+"_"+str(float('%.2f'%areaaboutdistance))+".ply",mesh,*mesh.get_attribute_names(),use_float=True,ascii=True)
    return tmp[2],float('%.2f' % areaaboutdistance),float('%.2f' % float(realdistance)) 

def dictsum(l1,l2):
    l = []
    for i in range(len(l1)-1):
        l.append(float(l1[i])+float(l2[i]))
    l.append(l1[-1]+1)
    return l

def dictchu(l1):
    l = []
    for i in range(len(l1)-1):
        l.append(float('%.2f'%(float(l1[i])/l1[-1])))
    return l

#in_fields = sys.argv[1].split("_")
#ply_id = in_fields[0]
#chain_id = in_fields[1]
#mesh = pymesh.load_mesh(SSPPIopts["ply_chain_dir"]+ply_id+"_"+chain_id+".ply")
#vertices = mesh.vertices
#resdiue_id = mesh.get_attribute("vertex_name")
#norm = mesh.get_attribute("vertex_normal")
#vert = {}

#for i in range(len(vertices)):
#    items = {i:str(norm[i*3+0])+" "+str(norm[i*3+1])+" "+str(norm[i*3+2])+"\t"+str(vertices[i][0])+" "+str(vertices[i][1])+" "+str(vertices[i][2])+"\t"+str(resdiue_id[i])}
#    vert.update(items)

#faces = mesh.faces
#centroid = mesh.get_attribute("face_centroid")
#normal = mesh.get_attribute("face_normal")
#circumcenter = mesh.get_attribute("face_circumcenter")

#for i in range(len(faces)):
#    for subface in faces[i]:
#        vert[subface] = vert[subface]+"\t"+ str(normal[i*3+0])+" "+str(normal[i*3+1])+" "+str(normal[i*3+2])+" "+str(centroid[i*3+0])+" "+str(centroid[i*3+1])+" "+str(centroid[i*3+2])+" "+str(circumcenter[i*3+0])+" "+str(circumcenter[i*3+1])+" "+str(circumcenter[i*3+2])

def compute_geo(vert):
    geometry_sum = {}
    for i in range(len(vert)):
        tmp = compute(vert[i],i)
        tmp_id = float(tmp[0])
        tmp_content = str(tmp[1:]).split('(')[1].split(')')[0].split(',')
        if tmp_id in geometry_sum:
            geometry_sum[tmp_id] = dictsum(geometry_sum[tmp_id],tmp_content)
        else:
            tmp_content.append(1)
            geometry_sum.update({tmp_id:tmp_content})
        sys.stdout.write(str('%.2f' % float(float(i) / float(len(vert)))) + '\r')
        sys.stdout.flush()
    print(kdtime)
    for key in geometry_sum.keys():
        geometry_sum[key] = dictchu(geometry_sum[key])
    print(len(vert))
    return geometry_sum

#compute_geo(vert)
#print(geometry_sum)
    
#print(compute(vert[50],50))
#print(compute(vert[10]))
