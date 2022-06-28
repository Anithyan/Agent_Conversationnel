import pandas as pd
import shutil
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json


""""

Ce fichier cherche à utiliser GASTnet pour convertir une animation 2D en 3D (keypoints json en entrée et sortie). Elle met donc en 
forme le JSON d'entrée pour correspondre au souhait de GASTnet, puis applique la reconstruction et enregistre le résultat en json.
( + Peut-être une visualisation 3D et vidéo)

"""

def convert_input(path):
    """
    2D to 3D AFTER THE AUTOENCODER

    """

    with open(path, 'r') as fr:
        video_info = json.load(fr)
    
    label = video_info["label"]
    label_index= 0
    le = len(video_info["data"])
    score = [1.0]
    bbox = [0,0,0,0] ### Gastnet do not use the bounding boxes nor the trusting score
    data = []
    for l in range(le):
        ske = {'pose': video_info["data"][l]["pose"],
        'score':len(video_info["data"][l]["pose"])*[score],
        'bbox':bbox
        }

        dictio = {'frame_index' : l+1,
        'skeleton':[ske.copy()]
        }
        data.append(dictio.copy())

    file = {
    'label' : label,
    'label_index': label_index,
    'data':data,
}

    with open(str(label)+'.json', 'w') as outfile:
        json.dump(file, outfile)


def apply_gast(path):
    """
    path = str(label)+'.json'
    
    """
    os.chdir(os.path.join(os.getcwd(),'./GAST-Net/'))
    os.system("python ./reconstruction.py -w ./27_frame_model.bin -n 17 -k ../"+str(path)+"  -vo ./output/test_npy.mp4 -kf mpii")
    os.chdir("../")

def visualisation(path) :
    """
    Visualisation to be made in 3D with plt, follow the visu made in GAST

    """


convert_input("D.json")

apply_gast("D.json")
