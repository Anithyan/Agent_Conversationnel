from pickle import TRUE
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json


#Load les data .npy issues des json en un modèle B25NB taille: n x 15 x 2
def load_data(name="kp_3.npy"):
    train_scenes = np.load(name,allow_pickle=True)

    size = 0
    x_train = []
    for scene in train_scenes :
        size += len(scene)
        for frame in scene :
            tmp = list(frame[0:30]) + list(frame[38:])
            x_train.append(tmp)  # remove the 4 head points

    x_train = np.array(x_train).reshape((size,42))
    x_train = np.array(x_train[:,:30]) # remove the 6 feet points


    x_train_x = x_train[:,::2]
    x_train_y = x_train[:,1::2]

    x_train = np.zeros((size,15,2))
    x_train[:,:,0] = x_train_x
    x_train[:,:,1] = x_train_y
    ### Shape de x_train [nb_frames_tot,15,2]
    return x_train

#normalisation des data entre -1 et 1 centrées en 0
def normalize(x_train):
    for f in range(len(x_train)) :
        M = np.max(x_train[f,:,1])
        m = np.min(x_train[f,:,1])
        x_train[f] = (2 * x_train[f]- M - m) / (M - m)
        x = x_train[f,8,0]
        x_train[f,:,0] -= x
    return(x_train)




#répresentation nuage de points 3D de l'espace latent avec la PCA
def latent_representation_PCA(x_train,show = TRUE):
    predict = (encoder.predict(x_train))
    tsne = PCA(n_components=2)
    X_embedded = tsne.fit_transform(predict)
    if show :
        plt.figure()
        plt.scatter(X_embedded[:,0],X_embedded[:,1])
        plt.show()
    return X_embedded, tsne

#interpolation ordre 1 entre 2 vecteurs
def interpolate_points(p1, p2, n=10):
	ratios = np.linspace(0, 1, num=n)
	vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	return np.asarray(vectors)

# réalise une PCA de dim n sur des data
def create_pca(data,components=3):
    pca = PCA(n_components=components)
    x = pca.fit_transform(data)
    
    print(pca.explained_variance_ratio_)
    return(pca,x)


#relie les points du B25NB en MPII compréhensible par GAST-Net
def new_binds(op):
    mpii = np.zeros(op.shape)
    L_mpii = [k for k in range(16)]
    L_op = [11,10,9,12,13,14,8,1,15,0,4,3,2,5,6,7]
    mpii[:,L_mpii] = op[:,L_op]
    return mpii

#Convertie les points B25NB en MPII compréhensible par GAST-Net
def to_Mpii(x_train):
    x = np.zeros((x_train.shape[0],16,2))
    x[:,:15,:] = x_train
    x = normalize(x)
    for f in range(len(x)):
        delta_y = x[f,1,1]- x[f,8,1]
        x[f,-1,0] = (x[f,1,1] - x[f,8,1])/delta_y * x[f,1,0]
        x[f,-1,1] = x[f,1,1] + delta_y/11.
        x[f,0,1] = x[f,-1,1] + delta_y*4/11.
        x[f,0,0] = (x[f,0,1] - x[f,8,1])/delta_y * x[f,-1,0]

    x = normalize(x)
    x = new_binds(x)
    return x

#nuage de point de l'esapce latent
def latent_representation(x_train):
    X = encoder.predict(x_train)
    plt.figure()
    plt.scatter(X[:,0],X[:,1])        
    plt.show()
    #for k in range (0, len(x_train)):

#choisis aléatoirement n poses dans la base de données, et réalise...
#  l'interpolation dans l'esapce latent puis récupère les poses associées...
# au format MPII dans un objet f
def anim_random(n=2):
    img = []
    coded_data = []
    pos = [np.random.randint(0,len(x_train)) for k in range(n)]
    for k in range(0,len(pos)-1) :
        test_image1=x_train[pos[k]].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
        test_image2=x_train[pos[k+1]].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
        encoded_img1=encoder.predict(test_image1)
        encoded_img2=encoder.predict(test_image2)
        interpolated_images=interpolate_points(encoded_img1.flatten(),encoded_img2.flatten(),n=30)
        #for i in interpolated_images :
        coded_data.append(pca.transform(interpolated_images))
        coded_data[k] = np.array(coded_data[k]).flatten()
        inter = interpolated_images.reshape(30,interpolated_images[:].shape[1])
        frames = (decoder.predict(inter))
        frames = to_Mpii(frames)
        img.append(frames) 
        
    f  = np.array(img).reshape((n-1)*30,16,2)

    return f,coded_data


def anim_json(anim,label="D"):
    """
    anim est un array de la forme [frame,sortie de l'ae]
    """
    
    le = len(anim)    

    data = []
    for l in range(le):
        ske = {'frame_index': l,
        'pose':(anim[l].tolist())
        }

        data.append(ske.copy())

    file = {
    'label' : label,
    'data':data,
    }

    with open(str(label)+'.json', 'w') as outfile:
        json.dump(file, outfile)


if __name__ == '__main__':
    autoencoder = tf.keras.models.load_model('./models/model_2')
    encoder = tf.keras.models.load_model('./models/model_2_encoder')
    decoder = tf.keras.models.load_model('./models/model_2_decoder')

    x_train = normalize(load_data(name="./data/kp_3.npy"))
    latent_size=2
    #latent_representation(x_train)
    
    pca,x = create_pca(encoder.predict(x_train))

    anim, coded_data = anim_random(n=3)
    anim_json(anim)
    print(np.shape(x), np.shape(coded_data))
    np.save("pca_xyz", x)
    np.save("coded_data",coded_data)
    