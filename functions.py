import pytube
import os

def read_videos(k=0):
    lines=[]
    f = open("./data/video_ids.txt", "r")
    i=0
    line=f.readline()
    while True:
        i+=1
        line=f.readline()
        if not line or i == k + 1 :
            break
        lines.append(line[:-1])
    return lines


def download_video(lines, n,filename):
    line = lines[n]
    url = 'https://www.youtube.com/watch?v=' + str(line)
    youtube = pytube.YouTube(url)
    youtube.streams.order_by('resolution').first().download(output_path='./videos/',filename=filename)
    return str(line)+'.3gpp'

def find_scenes(filename, path_to_folder,threshold=30.0):
    # Create our video & scene managers, then add the detector.
    os.system("scenedetect --input ../videos/"+filename+" -o ../scenes/scenes_first detect-content list-scenes save-images split-video")
