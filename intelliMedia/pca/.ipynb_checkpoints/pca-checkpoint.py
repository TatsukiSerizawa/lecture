import numpy as np
import pandas as pd 
 
all_data = pd.read_csv('wine.csv') 
 
 
col_num=all_data.shape[1]
 
#データインポート
X=all_data.iloc[1:,0:col_num-1]
Y=all_data.iloc[1:,col_num-1]
#features name
X_name=all_data.iloc[0,0:col_num-1]
 
 
 
#学習結果検証用　ratingがあるデータを、トレーニングデータと検証データに分解して利用。
from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion as Version
 
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split
 
X_train,X_test,Y_train,Y_test= train_test_split(
    X, Y, test_size=0.3, random_state=0)
 
#データの標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
 
X_train=np.array(X_train)
X_test=np.array(X_test)
Y_train=np.array(Y_train)
 
sc.fit(X_train)
X_train_std=sc.transform(X_train)
sc.fit(X_test)
X_test_std=sc.transform(X_test)
 
 
#PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
 
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
 
#==========================================================
#3 dimension
pca3 = PCA(n_components=3)
X_train_pca3 = pca3.fit_transform(X_train_std)
X_test_pca3 = pca3.transform(X_test_std)
 
 
 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
 
fig3 = plt.figure()
ax = Axes3D(fig3)
ax.scatter3D(np.ravel(X_train_pca3[Y_train==1,0]),np.ravel(X_train_pca3[Y_train==1,1]),np.ravel(X_train_pca3[Y_train==1,2]),c='r', marker='s',label='1')
ax.scatter3D(np.ravel(X_train_pca3[Y_train==2,0]),np.ravel(X_train_pca3[Y_train==2,1]),np.ravel(X_train_pca3[Y_train==2,2]),c='b', marker='x',label='2')
ax.scatter3D(np.ravel(X_train_pca3[Y_train==3,0]),np.ravel(X_train_pca3[Y_train==3,1]),np.ravel(X_train_pca3[Y_train==3,2]),c='g', marker='o',label='3')
ax.set_xlabel('pc 1')
ax.set_ylabel('pc 2')
ax.set_zlabel('pc 3')
plt.legend(loc='upper left')
 
plt.show()
#========================================================
#===========================================================
# 2 dimension
pca2 = PCA(n_components=2)
X_train_pca2 = pca2.fit_transform(X_train_std)
X_test_pca2 = pca2.transform(X_test_std)
 
fig2=plt.figure()
 
## 主成分の空間（平面）に射影
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
 
for l, c, m in zip(np.unique(Y_train), colors, markers):
    plt.scatter(X_train_pca2[Y_train == l, 0], 
                X_train_pca2[Y_train == l, 1], 
                c=c, label=l, marker=m)
 
#plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('pc 1')
plt.ylabel('pc 2')
plt.show()
 
#print('Covariance Matrix : %r' % pca.get_covariance())
print('Explained ratio : %r' % pca2.explained_variance_ratio_.round(2))
print('pc1 vector : %r' % pca2.components_[0].round(2))
print('pc2 vector : %r' % pca2.components_[1].round(2))
#==========================================================
 
 
#===========================================================
# 1 dimension
pca1 = PCA(n_components=1)
X_train_pca1 = pca1.fit_transform(X_train_std)
X_test_pca1 = pca1.transform(X_test_std)
 
fig0=plt.figure()
 
## 主成分の空間（平面）に射影
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
 
for l, c, m in zip(np.unique(Y_train), colors, markers):
    plt.scatter(X_train_pca1[Y_train == l], np.zeros(len(X_train_pca1[Y_train == l])),
                c=c, label=l, marker=m)
 
#plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('pc 1')
plt.ylabel('')
plt.show()
 
 
#==========================================================
#SVMを用いて分類
from sklearn.svm import SVC
 
####### apply non-linear svm ###################
## use "radial Basis Functional kernel" 動径基底カーネルを使用する。
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=1.0)
svm.fit(X_train_pca2, Y_train)
y_pred_svm=svm.predict(X_test_pca2)
 
result=y_pred_svm
 
  
from sklearn.metrics import accuracy_score
print('Accuracy : %.2f' % accuracy_score(np.rint(result), Y_test))
 
#==========================================================
#主成分分析の次元を変えて正答率を可視化
accuracy=[]
for i in range(1, 11):
 
    pca = PCA(n_components=i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
 
    svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=1.0)
    svm.fit(X_train_pca, Y_train)
    y_pred_svm=svm.predict(X_test_pca)
  
    from sklearn.metrics import accuracy_score
    #print('Accuracy : %.2f' % accuracy_score(np.rint(result), Y_test))
    accuracy.append(accuracy_score(y_pred_svm,Y_test))
 
fign=plt.figure()
plt.plot(range(1, 11), accuracy, marker='o')
plt.xlabel('n components')
plt.ylabel('accuracy')
#plt.savefig('./figures/elbow.png', dpi=300)
plt.show()