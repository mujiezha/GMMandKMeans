import numpy as np
import random
import math
#import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib import style
#from sklearn import mixture
#from matplotlib.colors import LogNorm
#style.use("ggplot")

fo = open("clusters.txt", "r");
points=[];
for line in fo:
    points.append([float(x) for x in line.split(',')])
#print (points)
#initial set
k=3
n=len(points)
dimen=len(points[0])
means=[] #initial means
means.append(random.randint(0,n-1))
condition=0 # the condition of how many points can change clusters
#initial means
while len(means)< k :
    temp=random.randint(0, n - 1)
    if temp not in means:
        means.append(temp)
#end initial

#k-means
clusterN=np.zeros((k,dimen)); #k cluster,n-dimension
clusterNsize=np.zeros(k);
meansN=np.zeros((k,dimen)); # means
clusterIndex=[None]*n;
change=n; #how many points are changing

for i in range(k):
    meansN[i]=points[means[i]]; # means

while change>condition:
    change=0;
    clusterN.fill(0);
    clusterNsize.fill(0);
    for point in range(n):
        belong=np.array([]);
        for mean in meansN:
            belong=np.append(belong,(np.sum((np.array(points[point])-np.array(mean))**2)));
        index=np.argmin(belong)
        clusterN[index]+=points[point]; #min-distance, new cluster
        clusterNsize[index] +=1; #clustersize
        if(clusterIndex[point]==None or index!=clusterIndex[point]):
            change+=1;
            clusterIndex[point]=index;

    for i in range(k):
        meansN[i]=np.divide(clusterN[i],clusterNsize[i]); #newmeans

print("K-Means result is")
print("K-Means centroid is (dimension * cluster)\n",np.transpose(meansN))
print("\n")

#end of K-means

#GMM
#initialize
PointMatrx=np.array(points);
PointMatrx=np.transpose(PointMatrx);
# (dimension,n)
Gcluster=clusterIndex;
transU=np.zeros((dimen,k));
UofD=np.zeros(k);
Covar=np.zeros((k,dimen,dimen));
Ric=np.zeros((n,k));

for i in range(n):
    Ric[i][Gcluster[i]]=1;

Nk=Ric.sum(axis=0)
Tcc=np.divide(Nk,n)
change=n;
condition=5
#end initialize

while change > condition:
    change=0;
    transU.fill(0)
    Covar.fill(0)
    # for transU
    transU=np.dot(PointMatrx,Ric);
    for i in range(k):
        transU[:,i]=np.divide(transU[:,i],Nk[i])
    # for Covar
    for i in range(n):
        pm = np.reshape(PointMatrx[:, i], [dimen, 1])
        for c in range(k):
            trans = transU[:, c];
        #print(np.dot((pm - trans), np.transpose(pm - trans)))
            Covar[c] +=np.divide(np.multiply(np.dot((pm - trans), np.transpose(pm - trans)), Ric[i][c]),Nk[c]);
    # for each Ric
    for i in range(n):
        belong = np.zeros(k);
        for c in range(k):
            pm = np.reshape(PointMatrx[:, i], [dimen, 1])
            trans=np.reshape(transU[:, c], [dimen, 1])
            temp = np.dot(np.transpose(pm - trans), np.linalg.inv(Covar[c]));
            temp = np.dot(temp, pm - trans)
            bel=Tcc[c]*math.exp(-0.5 * temp) * math.pow(np.linalg.det(Covar[c]), -0.5) * math.pow(2 * math.pi,
                                                                                            -0.5 * dimen);
            belong[c] = bel;
        sum = np.sum(belong);
        belong=np.divide(belong, sum)
        for c in range(k):
            Ric[i][c] =belong[c];
        if Gcluster[i]!=np.argmax(belong):
            change+=1;
            Gcluster[i]= np.argmax(belong)
    Nk = Ric.sum(axis=0)
    Tcc = np.divide(Nk, n)
    #print(Tcc)
    #print(Ric)

    #print(Gcluster)
    #print(change)
    #print(Tcc)
    #print(Covar)
    #print(transU)
    #print(Tcc)
print("GMM result is")
print("GMM mean is (dimension * cluster) \n",transU)
print("amplitude is\n ",Tcc)
print("covariance is\n",Covar)


#clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
#clf.fit(np.transpose(PointMatrx))
#mean, amplitude and covariance matrix

##






