from statistics import mean
from functions import *
import pandas as pd


def extract_data(lines,begin,end):
    for n in range(begin,end):
        filename = str(n)+".3gpp"
        download_video(lines,n-1,filename)
        os.chdir(os.path.join(os.getcwd(),'./openpose/'))
        find_scenes(filename,0)

        for scene in os.listdir(os.path.join(os.getcwd(),'../scenes/scenes_first')):
            if scene[-4:] == ".mp4":
                os.mkdir("../valid_outputs/"+str(scene))
                os.system('bin\OpenPoseDemo.exe --video ../scenes/scenes_first/'+(scene)+' --net_resolution -1x320 --display 0 --write_json ../output/result --write_video ../valid_outputs/'+str(scene)+"/"+'0.avi')
                os.remove('../scenes/scenes_first/'+scene)

                for file in os.listdir("../output/result"):
                    df = pd.read_json("../output/result/"+file)

                    if (len(df["people"]) == 1) :
                        l = df["people"][0]["pose_keypoints_2d"][0:45]+df["people"][0]["pose_keypoints_2d"][57:]
                        confidence = sorted(df["people"][0]["pose_keypoints_2d"][2:45:3]+df["people"][0]["pose_keypoints_2d"][59::3])
                        not_sure = (mean(confidence[:5]) < 0.5)
                    else :
                        l=[]
                        not_sure = False  

                    if len(df["people"]) != 1 or (0 in l) or not_sure:
                        os.remove("../output/result/"+file)
                    else :                
                        os.rename('../output/result/'+file, "../valid_outputs/"+str(scene)+"/"+file)
                
            else:
                os.remove('../scenes/scenes_first/'+scene)
        os.remove('../videos/'+str(filename))
        os.chdir('../')

def remove_non_useful_folders():
    for scene in os.listdir(os.path.join(os.getcwd(),'./valid_outputs/')):
        if len(os.listdir(os.path.join(os.getcwd(),'./valid_outputs/'+scene))) == 1 :
            os.remove("./valid_outputs/"+str(scene)+"/0.avi")
            os.rmdir(os.path.join(os.getcwd(),'./valid_outputs/'+scene))



if __name__ == '__main__':

    lines = read_videos(k=150)
    print(lines)
    extract_data(lines,begin=77,end=140)
    remove_non_useful_folders()