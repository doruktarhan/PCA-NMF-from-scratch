import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = np.load("faces_data.npy")



# PART 3A
plt.figure(1)
plt.suptitle("6 sample faces")  
for i in range(6):
    image_sample = data[i].reshape(64,-1)
    plt.subplot(2,3,i+1)
    plt.imshow(image_sample, cmap = 'gray')
  
    
# PART 3B
mean_face = np.mean(data, axis = 0)
plt.figure(2)
plt.imshow(mean_face.reshape(64,-1), cmap = 'gray')
plt.title("mean face")

plt.figure(3)
plt.suptitle("6 sample faces centralized")
for i in range(6):
    image_sample = (data[i]-mean_face).reshape(64,-1)
    plt.subplot(2,3,i+1)
    plt.imshow(image_sample, cmap = 'gray')


#PART 3C
centralized_data = data - mean_face
cov_matrix = np.cov(centralized_data,rowvar = False) 


eig_val, eig_vec = np.linalg.eigh(cov_matrix)
sorted_eig  = np.argsort(-eig_val)
eig_val = eig_val[sorted_eig]
eig_vec = eig_vec[:, sorted_eig]


total_var = np.sum(eig_val) / (len(centralized_data)-1)
print("Total variance: ",total_var)

exp_var = (eig_val/total_var)/(len(centralized_data)-1)

#exp_var = np.flip(exp_var) 

count = 0
for i in range(np.size(exp_var)-1):
    exp_var[i+1] = exp_var[i] + exp_var[i+1]
    if exp_var[i] > 0.90 and count == 0:
        count = i

print("The required number of PCs to preserve at least 90% of variation in the data is ",count)
    

plt.figure(4)
plt.plot(exp_var)
plt.title("explained variance vs number of PC")
plt.xlabel("Number of PC")
plt.ylabel("explained variance")




#PART 3D
U,S,VT = np.linalg.svd(centralized_data.T)
plt.figure(13)
plt.suptitle("First 64 EigenFaces")
for i in range (64):
    pc_sample = U[:,i].reshape(64,-1)
    plt.subplot(8,8,i+1)
    plt.imshow(pc_sample, cmap = 'gray')

PC_max = U[:,:64] #first 64 PC
reduced_data = centralized_data @ PC_max
img_1_scores = reduced_data[0].reshape(8,8)
plt.figure(5)
sns.heatmap(img_1_scores,cbar = True)
plt.title("image 1 mixing coeffcients via SVD")



#PART 3E
reconstructed_images = reduced_data @ PC_max.T 
reconstructed_images = reconstructed_images + mean_face
plt.figure(6)
plt.suptitle("Reconstructed images SVD")  
for i in range(6):
    image_sample = reconstructed_images[i].reshape(64,-1)
    plt.subplot(2,3,i+1)
    plt.imshow(image_sample, cmap = 'gray')

#PART 3F 

def NMF(data, scale_fact,K):
    #create postive nonnegative matices W and H
    W = abs(np.random.normal(size = (400,K))*scale_fact) #400xK
    H = abs(np.random.normal(size = (K,4096))*scale_fact)#K*4096
    
    
    #update H and W
    condition = True
    i = 1
    while condition:
        H = H*(np.divide((W.T @ data),(W.T@W@H))) 
        W = W*(np.divide((data@H.T),(W@H@H.T))) 
        error = np.linalg.norm(data-W@H)
        error_list.append(error)
        
        #convergence condition
        if len(error_list) > 1:
            if (error_list[-2]-error_list[-1])/error_list[-1] < 0.0001:
                condition = False
    
    return W,H,error_list




#PART 3G

error_list = []
K = 64
mean_data = np.mean(data)
scale_fact = np.sqrt(mean_data/400)




#find W and H with NMF
W,H,error_list = NMF(data,scale_fact,K)

#plot loss fnction
plt.figure(7)
plt.plot(error_list)
plt.title("Loss History via NMF")
plt.ylabel("Loss")
plt.xlabel("# of iterations")


plt.figure(8)
#plot eigenfaces
plt.suptitle(" 64 Eigenfaces via NMF")
for i in range (64):
    pc_sample = H[i].reshape(64,-1)
    plt.subplot(8,8,i+1)
    plt.imshow(pc_sample, cmap = 'gray')
    
#plot the mixing coefficient of first person
plt.figure(9)
sns.heatmap(W[0].reshape(8,8),cbar= True)
plt.title("image 1 mixing coeffcients via NMF")





#PART 3H
reconstructed_images_NMF = W @ H
plt.figure(10)
plt.suptitle("Reconstructed images via NMF")  
for i in range(6):
    image_sample = reconstructed_images_NMF[i].reshape(64,-1)
    plt.subplot(2,3,i+1)
    plt.imshow(image_sample, cmap = 'gray')
    
    
    
#PART 3E and 3H error faces    
error_faces_NMF = data[:6,:] - reconstructed_images_NMF[:6,:]
plt.figure(11)
plt.suptitle("Error images via NMF")  
for i in range(6):
    image_sample = error_faces_NMF[i].reshape(64,-1)
    plt.subplot(2,3,i+1)
    plt.imshow(image_sample, cmap = 'gray')



error_faces = data[:6,:] - reconstructed_images[:6,:]
plt.figure(12)
plt.suptitle("Error images via SVD")  
for i in range(6):
    image_sample = error_faces[i].reshape(64,-1)
    plt.subplot(2,3,i+1)
    plt.imshow(image_sample, cmap = 'gray')
    