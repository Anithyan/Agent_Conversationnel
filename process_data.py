import pandas as pd
import os
import numpy as np


def create_npy(name,path="./valid_outputs/",scene_len = None):
    """
    scene_len == 'max', entier naturel ou autre (ex : None)

    """
    if scene_len == "max":
        maximum = 0
        for scene in os.listdir(path):
            length = np.max(len(os.listdir(path+scene))-1)
            if maximum <  length:
                maximum = length
        if maximum%2 == 1 :
            maximum +=1
    elif type(scene_len) == int :
        maximum = scene_len
        if maximum%2 == 1 :
            maximum +=1
    else :
        maximum = -1

    kp_total = []
    for scene in os.listdir(path):
        kp_scene=[]
        for frame in os.listdir(path+scene):
            if frame[-4:]!='.avi' and len(os.listdir(path+scene)) > 0 :
                df = pd.read_json(path+scene+"/"+frame)
                people = df["people"]
                kp = people[0]["pose_keypoints_2d"]
                kp = np.delete(kp,[i for i in range(2,len(kp),3)])
                kp_scene.append(kp)

        if maximum >= 0 :
            temp = np.zeros((maximum,50))
            temp[:len(kp_scene)] = kp_scene
            kp_scene = temp
        
        if np.size(kp_scene) > 0:
            kp_total.append(np.array(kp_scene))
    np.save(name,np.array(kp_total,dtype=object))


if __name__ == '__main__':
    create_npy("./data/kp",scene_len=None)
