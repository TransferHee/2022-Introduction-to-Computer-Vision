import numpy as np
# from sklearn.cluster import KMeans

'''
2018314848 한승희
Introduction to Computer Vision
Assignment #4
env:
    Python 3.7.9
    numpy 1.12.6
'''

def VLAD(visual_dict_1, centers, labels, N=1000, D=1024):
    des = np.zeros((N, D))
    vd_1_idx = 0
    for n in range(len(visual_dict_1)):
        for label in range(8):
            label_idx = np.where(labels[vd_1_idx:vd_1_idx+len(visual_dict_1[n])]==label)[0]
            all_residual = np.sum(visual_dict_1[n][label_idx] - centers[label], axis=0)
            
            # des[n, label*128:(label+1)*128] = all_residual
            # des[n, label*128:(label+1)*128] = np.sign(all_residual) * np.log(np.abs(all_residual)+0.1)
            des[n, label*128:(label+1)*128] = np.sign(all_residual) * np.sqrt(np.abs(all_residual))
            # des[n, label*128:(label+1)*128] = np.sign(all_residual) * np.cbrt(np.abs(all_residual))
            # des[n, label*128:(label+1)*128] = all_residual/sum(np.sqrt(all_residual**2))

        # l1 norm
        des[n] /= np.linalg.norm(des[n], 1)

        # l2 norm
        # des[n] /= np.linalg.norm(des[n], 2)

        # infinity-norm
        # des[n] /= np.linalg.norm(des[n], np.inf)

        vd_1_idx += len(visual_dict_1[n])
    
    # binary format file
    with open('A4_2018314848.des', 'wb') as f:
        f.write(np.array([N, D], dtype=np.int32).tobytes())
        f.write(des.astype(np.float32).tobytes())

if __name__ == "__main__":

    # 두가지 형석의 visual dictionary 생성
    visual_dict_1 = []
    for i in range(1000):
        with open(f'./feats/{i:0>5}.sift', 'rb') as f:
            visual_dict_1.append(np.fromfile(f, dtype=np.uint8).reshape(-1, 128))
    visual_dict_1 = np.array(visual_dict_1)

    visual_dict_2 = []
    for sift in visual_dict_1:
        for v in sift:
            visual_dict_2.append(v)
    visual_dict_2 = np.array(visual_dict_2)
    print('make visual dictionary from sift files')

    # 여기서부턴 save & load 형식으로 변경!

    # 1024 // 128 = 8 이므로 최대 8개 클러스터 생성
    # cluster = KMeans(n_clusters=8, random_state = 777)
    # cluster.fit(visual_dict_2)

    # labels = cluster.predict(visual_dict_2)
    # centers = cluster.cluster_centers_
    # np.save('centers',centers)
    # np.save('labels', labels)

    centers = np.load("centers.npy")
    labels = np.load("labels.npy")
    print('get centers and labels')

    VLAD(visual_dict_1, centers, labels)
    print('finish VLAD!')
    # eval results
    # L1: 3.3220 / L2: 3.3210
    # Accuracy: 3.32200