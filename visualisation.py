import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import imageio
from tools_ae import *


#viualisation des squelettes, format B25NB ("own") ou MPII
def visu_skel(x_train,index,format = "own"):
    frame = x_train[index]
    
    plt.figure()
    plt.xlim(left = -1.1,right=1.1)
    plt.ylim(top=1.1,bottom=-1.1)
    plt.scatter(frame[:,0],-frame[:,1])

    if format == "own":
        L = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[1,11],[8,9],[9,10],[11,12],[12,13]]
    elif format == "mpii":
        L = [[0,1],[1,2],[2,6],[3,6],[3,4],[4,5],[6,7],[7,8],[8,9],[7,12],[10,11],[11,12],[17,13],[13,14],[14,15]]    

    for k in L:
        plt.plot([frame[k[0],0],frame[k[1],0]],[-frame[k[0],1],-frame[k[1],1]])
    plt.show()

def latent_space_2_skel():
    print( decoder.predict([[0,0]]))
    
#affiche un subplot 2 x 9 avec l'encodage des interpolations dans l'espace latent (en haut) et dans l'espace 3D (en bas)
def visu_interpo(x_train,latent_size=2):
    test_image1=x_train[np.random.randint(0,len(x_train))].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
    test_image2=x_train[np.random.randint(0,len(x_train))].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
    encoded_img1=encoder.predict(test_image1)
    encoded_img2=encoder.predict(test_image2)
    interpolated_images=interpolate_points(encoded_img1.flatten(),encoded_img2.flatten())
    interpolated_orig_images=interpolate_points(test_image1.flatten(),test_image2.flatten())
    predict = (encoder.predict(x_train[::50]))

    if latent_size > 2 :
        pca,x = create_pca(predict)
        inter = pca.transform(interpolated_images)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:,0],x[:,1],x[:,2],s=5,label="X_train")
        ax.scatter(inter[:,0],inter[:,1],inter[:,2],label="Interpolation")
        ax.scatter(inter[0,0],inter[0,1],inter[0,2],label="Beginning")
        ax.scatter(inter[-1,0],inter[-1,1],inter[-1,2],label="End")
        ax.legend()
        plt.show()

    else : 
        plt.figure()
        plt.scatter(predict[:,0],predict[:,1],label="X_train")
        plt.scatter(interpolated_images[:,0],interpolated_images[:,1],label="Interpolation")
        plt.scatter(interpolated_images[0,0],interpolated_images[0,1],label="Beginning")
        plt.scatter(interpolated_images[-1,0],interpolated_images[-1,1],label="End")
        plt.legend()
        plt.show()


    if format == "own":
        L = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[1,11],[8,9],[9,10],[11,12],[12,13]]
    elif format == "mpii":
        L = [[0,1],[1,2],[2,6],[3,6],[3,4],[4,5],[6,7],[7,8],[8,9],[7,12],[10,11],[11,12],[17,13],[13,14],[14,15]]    



    interpolated_images.shape
    num_images = 10
    np.random.seed(42)
    plt.figure(figsize=(20, 8))

    for i, image_idx in enumerate(interpolated_images):
        
        ax = plt.subplot(5, num_images,num_images+ i + 1)
        inter = interpolated_images[i].reshape(1,interpolated_images[i].shape[0])
        frame = decoder.predict(inter)[0]
        
        plt.scatter(frame[:,0],-frame[:,1],s=10)
        for k in L:
            plt.plot([frame[k[0],0],frame[k[1],0]],[-frame[k[0],1],-frame[k[1],1]])

        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(left = -1.1,right=1.1)
        ax.set_ylim(top=1.1,bottom=-1.1)
        ax.set_title("Latent: {}".format(i))
        
        ax = plt.subplot(5, num_images,2*num_images+ i + 1)
        frame = interpolated_orig_images[i]
        plt.scatter(frame[0::2],-frame[1::2],s=10)
        for k in L:
            plt.plot([frame[2*k[0]],frame[2*k[1]]],[-frame[2*k[0]+1],-frame[2*k[1]+1]])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(left = -1.1,right=1.1)
        ax.set_ylim(top=1.1,bottom=-1.1)
        ax.set_title("Image: {}".format(i))

    plt.show()


#affiche le squelette en sortie du codeur en ayant encodé xtrain[index]
def simple_visu(x_train,index):
    test_image=x_train[index].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
    encoded_img1=autoencoder.predict(test_image)
    visu_skel(encoded_img1,0)


def chose_point(x_train,random=True,n=1,step=50):
    data = (encoder.predict(x_train[::step]))
    pca,x = create_pca(data)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:,0],x[:,1],x[:,2],s=5,label="X_train")
    ax.legend()
    plt.show()

    for k in range(n):
        if random :
            point_1 = (np.max(x[:,0])- np.min(x[:,0])) * np.random.random() - np.min(x[:,0])
            point_2 = (np.max(x[:,1])- np.min(x[:,1])) * np.random.random() - np.min(x[:,1])
            point_3 = (np.max(x[:,2])- np.min(x[:,2])) * np.random.random() - np.min(x[:,2])
        else :
            print("x : ")
            point_1 = input()
            print("y : ")
            point_2 = input()
            print("z : ")
            point_3 = input()

        point = pca.inverse_transform([float(point_1),float(point_2),float(point_3)])
        constr = decoder.predict(np.array([point]))
        visu_skel(constr,0,format="own")





#créé un gif avec une interpo entre 10 poses
def create_gif(format = "own",name = 'mygif.gif'):
    if format =='own':
        L = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[1,11],[8,9],[9,10],[11,12],[12,13]]
    elif format == 'mpii':
        L = [[0,1],[1,2],[2,6],[3,6],[3,4],[4,5],[6,7],[7,8],[8,9],[7,12],[10,11],[11,12],[17,13],[13,14],[14,15]]    

    pos = [np.random.randint(0,len(x_train)) for k in range(10)]
    i=0
    files = []
    for k in range(0,len(pos)-1) :
        test_image1=x_train[pos[k]].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
        test_image2=x_train[pos[k+1]].reshape(1,x_train[0]. shape[0],x_train[0].shape[1])
        encoded_img1=encoder.predict(test_image1)
        encoded_img2=encoder.predict(test_image2)
        interpolated_images=interpolate_points(encoded_img1.flatten(),encoded_img2.flatten(),n=30)


        inter = interpolated_images.reshape(30,interpolated_images[:].shape[1])
        frames = (decoder.predict(inter))

        filenames = []
        
        for frame in frames:
            i+=1
            plt.scatter(frame[:,0],-frame[:,1],s=10)
            for k in L:
                plt.plot([frame[k[0],0],frame[k[1],0]],[-frame[k[0],1],-frame[k[1],1]])
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            
            # create file name and append it to a list
            filename = f'{i}.png'
            filenames.append(filename)
            files.append(filename)
            # save frame
            plt.savefig(filename)
            plt.close()
    # build gif
    with imageio.get_writer(name, mode='I') as writer:
        for filename in files:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(files):
        os.remove(filename)


if __name__ == '__main__':

    autoencoder = tf.keras.models.load_model('./models/model_2')
    encoder = tf.keras.models.load_model('./models/model_2_encoder')
    decoder = tf.keras.models.load_model('./models/model_2_decoder')

    latent_space_2_skel()
    #x_train = normalize(load_data(name="kp_3.npy"))
    #latent_representation(x_train)
    #simple_visu(x_train, 1000)
    #visu_interpo(x_train,latent_size=latent_size)
    #chose_point(x_train,random=True,n=5)
    #create_anim()
    # visu_interpo(x_train,latent_size=latent_size)
    # create_gif()