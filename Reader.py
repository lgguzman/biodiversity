import numpy as np
from matplotlib.mlab import PCA
from Diversity import DiversityAlpha as DA
from Diversity import  DiversityBeta as DB


class Reader:

    def __init__(self, otupath, muestrapath, taxonomypath):
        self.readOtu(otupath)
        self.readMuestra(muestrapath)
        self.readTaxonomy(taxonomypath)
        self.otusSimpleType()
        self.otusReduced()

    def readOtu (self, otupath=None):
        if otupath != None:
            self.otupath = otupath
        with open(self.otupath, 'r') as f:
                temp = [x.strip().replace('\"', "").replace('\'', "").split('\t') for x in f]
                self.otu = [[int(x) for x in y[1:]] for y in temp[1:]]
                self.oturows = [row[0] for row in temp[1:]];
                self.otuheader = temp[0][1:]

    def readMuestra (self, muestrapath=None):
        if muestrapath != None:
            self.muestrapath = muestrapath
        with open(self.muestrapath, 'r') as f:
                self.muestra = [x.strip().replace('\"', "").replace('\'', "").split('\t') for x in f]

    def readTaxonomy (self, taxonomypath=None):
        if taxonomypath != None:
            self.taxonomypath = taxonomypath
        with open(self.taxonomypath, 'r') as f:
                temp = [x.strip().replace('\"', "").replace('\'', "").split('\t') for x in f]
                self.taxonomia = [[x for x in y] for y in temp[1:]]

    def otusSimpleType(self):
        self.otusType = [None]*len(self.otuheader)
        column= [row[0] for row in self.muestra]
        for i in range(0, len(self.otuheader)):
            self.otusType[i]= self.muestra[column.index(self.otuheader[i])][5]

    def otusReduced(self):
        sortedtaxonomy = np.array(sorted(self.taxonomia, key=lambda tax: tax[5]))
        data= np.array(self.otu)
        nrows=len(self.taxonomia)
        ncolumn = len(self.otu[0])
        temp = np.zeros(ncolumn)
        self.reducedOtu = None
        tempName = sortedtaxonomy[0,5]
        list=[]
        textoTotal = ''
        texto = []
        textoNA=''
        textoFamilia=  'Familia ' + tempName + ':'
        for i in range (nrows):
            if sortedtaxonomy[i, 5] != 'NA':
                if tempName == sortedtaxonomy[i, 5]:
                    texto.append(sortedtaxonomy[i, 0])
                    temp += data[self.oturows.index(sortedtaxonomy[i, 0])]
                else:
                    if(len(texto)>1 ):
                        textoTotal = textoTotal + '\n'+textoFamilia+'\n'
                        for k in range(len(texto)):
                            textoTotal += texto[k] + ', '
                        texto = []
                    tempName = sortedtaxonomy[i, 5]
                    list.append(temp)
                    temp = data[self.oturows.index(sortedtaxonomy[i, 0])]
                    textoFamilia = 'Familia ' + tempName + ' :'
                    texto.append(sortedtaxonomy[i, 0])
            else:
                textoNA += sortedtaxonomy[i, 0] +  ', '
        list.append(temp)
        self.reducedOtu = np.stack(list, axis=0)
        self.reducedOtu = self.reducedOtu.tolist()


data =  Reader('otu.txt','muestra.txt','taxonomia.txt');
divB =  DB()
distance = divB.BrayCurtis(data.reducedOtu)[0]
x = (divB.NMMS(distance, alpha=3, iteration=200))
divB.plot(data.otusType,x)
# distance = [[0.0, 3.0, 2.0, 5.0], [3.0, 0.0, 1.0, 4.0], [2.0, 1.0, 0.0, 6.0], [5.0, 4.0, 6.0, 0.0]]
# divB =  DB()
# #xini = np.array( divB.PCoA(distance)[0])
# xini = np.array([[ 3.0, 2], [ 2, 7], [ 1,  3], [10,  4]])
# n = len(xini)
# m = int(n*(n+1)/2-n)
# matrix=  [[0] * 8 for i in range(m)]
# k=1
# i=0
# for k in range(n):
#     for j in range(k+1,n):
#         matrix[i][0] = k
#         matrix[i][1] = j
#         matrix[i][2] = distance[matrix[i][0]][matrix[i][1]]
#         matrix[i][3] = np.linalg.norm(xini[matrix[i][0]]-xini[matrix[i][1]])
#         i=i+1
#     k=k+1
# matrix=np.array(matrix)
# matrix=matrix[matrix[:,2].argsort()]
# sw=True
# while(sw):
#     i=0
#     while (i<m):
#         temp = matrix[0][3]
#         sum = matrix[i][3]
#         count=1
#         pp = sum / count
#         while(i<m-1 and pp > matrix[i+1][3]):
#             sum=sum+matrix[i+1][3]
#             count=count+1
#             pp = sum / count
#             i=i+1
#         while (count>0):
#             matrix[i-count+1][4]=pp
#             count=count-1
#         i=i+1
#     for i in range(m):
#         matrix[i][5] = (matrix[i][3]-matrix[i][4])**2
#         matrix[i][6] = matrix[i][3] * matrix[i][3]
#         matrix[i][7] = ( matrix[i][3]-np.mean(matrix[:,3]))**2
#     if ( np.sqrt((matrix[:,5]).sum()/(matrix[:,7]).sum()) < 0.001):
#         sw=False
#     alpha = 3
#     for i in range (n):
#         sum=0
#         sum2 = 0
#         for j in range (n):
#             if (i!=j):
#                 for k in range(m):
#                     if (matrix[k][0] == i and matrix[k][1] == j) or (matrix[k][0] == j and matrix[k][1] == i):
#                         sum=sum+ (1-matrix[k][4]/matrix[k][3])*(xini[j][0]-xini[i][0])
#                         sum2 = sum2 + (1 - matrix[k][4] / matrix[k][3]) * (xini[j][1] - xini[i][1])
#         xini[i][0]= float(xini[i][0]+(alpha/(n-1))*sum)
#         xini[i][1] = float(xini[i][1] + (alpha / (n - 1)) * sum2)
#     print(xini)
#     for i in range (m):
#         matrix[i][3] = np.linalg.norm(xini[int(matrix[i][0])]-xini[int(matrix[i][1])])