from skbio.diversity import alpha
from skbio.diversity import get_beta_diversity_metrics
from skbio.stats.ordination import pcoa as Pcoa
from skbio.stats.distance import permanova
from skbio.stats.distance import DistanceMatrix
from sklearn.decomposition import PCA as sklearnPCA
from scipy import linalg as LA
import matplotlib.pyplot as pp
import numpy as np
import math


class DiversityAlpha:
    def chao1(self, otu, bias_corrected=False):
        diversity = [0] * len(otu[0])
        for j in range(len(otu[0])):
            columnj = [row[j] for row in otu]
            f2 = len([x for x in columnj if x == 2])
            f1 = len([x for x in columnj if x == 1])
            s = len([x for x in columnj if x > 0])
            if not bias_corrected:
                diversity[j] = s + f1 * (f1 - 1) / 2 if f2 == 0 else f1 * f1 / (2 * f2)
            else:
                diversity[j] = s + f1 * (f1 - 1) / (2 * (f2 + 1))
        return diversity

    def testChao1(self, otu):
        diversity = [0] * len(otu[0])
        for j in range(len(otu[0])):
            diversity[j] = alpha.chao1([row[j] for row in otu], bias_corrected=True)
        print(diversity)
        print(self.chao1(otu, bias_corrected=True))

    def shannon(self, otu):
        diversity = [0] * len(otu[0])
        for j in range(len(otu[0])):
            s = [x for x in [row[j] for row in otu] if x > 0]
            ssum = sum(s);
            p = [x / ssum for x in s]
            diversity[j] = -1 * sum([x * math.log(x, 2) for x in p])
        return diversity

    def simpson(self, otu):
        diversity = [0] * len(otu[0])
        for j in range(len(otu[0])):
            s = [x for x in [row[j] for row in otu] if x > 0]
            ssum = sum(s);
            p = [x / ssum for x in s]
            diversity[j] = 1 - sum([x * x for x in p])
        return diversity

    def testShannon(self, otu):
        diversity = [0] * len(otu[0])
        for j in range(len(otu[0])):
            diversity[j] = alpha.shannon([row[j] for row in otu])
        print(diversity)
        print(self.shannon(otu))

    def testSimpson(self, otu):
        diversity = [0] * len(otu[0])
        for j in range(len(otu[0])):
            diversity[j] = alpha.simpson([row[j] for row in otu])
        print(diversity)
        print(self.simpson(otu))

    def plot(self, headers, points, color='r'):
        x = np.array(range(0, len(headers)))
        y = np.array(points);
        my_xticks = headers;
        pp.xticks(x, my_xticks, rotation=90)
        pp.plot(x, y, color + '.');
        pp.show();

    def plot2(self, headers, points, name):
        group = {}
        for i in range(len(points)):
            if not (headers[i] in group):
                group[headers[i]] = []
            group[headers[i]].append(points[i])
        headers = []
        shannon = [[] for j in range(3)]
        for key, value in group.items():
            headers.append(key)
            for k in range(3):
                shannon[k].append(value[min(len(value) - 1, k)])
        x = np.array(range(0, len(headers)))
        my_xticks = headers;
        pp.xticks(x, my_xticks, rotation=90)
        y1 = np.array(shannon);
        line1 = []
        labels = []
        temp, = pp.plot(x, y1[0], 'r.');
        temp, = pp.plot(x, y1[1], 'r.');
        temp, = pp.plot(x, y1[2], 'r.');
        line1.append(temp)
        labels.append(name)
        pp.legend(line1, labels)
        pp.show();

    def plotComparission(self, shannon, chao1, simpson):
        shannon=np.array(shannon)
        shannon = shannon/ np.linalg.norm(shannon)
        chao1 = np.array(chao1)
        chao1 = chao1 / np.linalg.norm(chao1)
        simpson = np.array(simpson)
        simpson = simpson / np.linalg.norm(simpson)
        fig, axes = pp.subplots(nrows=1, ncols=2, figsize=(9, 4))
        headers = ['Shannon', 'Chao1', 'Simpson']
        all_data = [shannon, chao1, simpson]
        axes[0].violinplot(all_data,
                           showmeans=False,
                           showmedians=True)
        axes[0].set_title('Violin')
        axes[1].boxplot(all_data)
        axes[1].set_title('Box')
        for ax in axes:
            ax.yaxis.grid(True)
            ax.set_xticks([y + 1 for y in range(len(all_data))])
            ax.set_xticklabels(headers, rotation=90)
            ax.set_xlabel('Diversity Index')
            ax.set_ylabel('Diversity')
        pp.show()


    def plotViolin(self,headers,points,title):
        fig, axes = pp.subplots(nrows=1, ncols=2, figsize=(9, 4))
        group = {}
        for i in range(len(points)):
            if not (headers[i] in group):
                group[headers[i]] = []
            group[headers[i]].append(points[i])
        headers = []
        i=0
        all_data=[]
        for key, value in group.items():
            headers.append(key)
            all_data.append(np.array(value))
            i=i+1
        axes[0].violinplot(all_data,
                           showmeans=False,
                           showmedians=True)
        axes[0].set_title('Violin')
        axes[1].boxplot(all_data)
        axes[1].set_title('Vox')
        for ax in axes:
            ax.yaxis.grid(True)
            ax.set_xticks([y + 1 for y in range(len(all_data))])
            ax.set_xticklabels(headers,rotation=90)
            ax.set_xlabel('Types')
            ax.set_ylabel('Diversity')

        fig.suptitle(title)
        pp.show()

    def list(self):
        print(get_beta_diversity_metrics());


class DiversityBeta:
    def BrayCurtis(self, otu):
        data = np.array(otu)
        n = len(otu[0])
        diversity = [[(np.absolute(data[:, i] - data[:, j])).sum() / (data[:, i] + data[:, j]).sum() for i in range(n)]
                     for j in range(n)]
        return diversity, [[1 - x for x in row] for row in diversity]

    def Canberra(self, otu):
        data = np.array(otu)
        n = len(otu[0])
        diversity = [[(self.div0(np.absolute(data[:, i] - data[:, j]), (np.absolute(data[:, i]) + np.absolute(data[:, j])))).sum() for i in range(n)]
                     for j in range(n)]
        return diversity, [[1 - x for x in row] for row in diversity]

    def div0(self, a, b):
        """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~ np.isfinite(c)] = 0  # -inf inf NaN
        return c

    def LCorrection(self, otu):
        data =np.array(otu)
        return (data + 1).tolist()

    def Jaccard(self, otu):
        data = np.array(otu)
        n = len(otu[0])
        diversity = [
            [ np.minimum(data[:,i], data[:,j]).sum()/np.maximum(data[:,i], data[:,j]).sum() for i in
             range(n)] for j in range(n)]
        return  [[1 - x for x in row] for row in diversity], diversity

    # Image Processing Book
    def PCA(self, distance, dims_rescaled_data=2):
        data = np.array(distance)
        data -= data.mean(axis=0)
        R = np.cov(data, rowvar=False)
        evals, evecs = LA.eigh(R)
        closeZero = np.isclose(evals, 0)
        evals[closeZero] = 0
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]
        evecs = evecs[:, :dims_rescaled_data]
        return np.dot(evecs.T, data.T).T, evals, evecs

    # https://sites.google.com/site/analisismultivariados/coordenadas-principales, https://en.wikipedia.org/wiki/Multidimensional_scaling#Types
    def PCoA(self, distance, dims_rescaled_data=2):
        data = np.array(distance)
        data = data * data / -2
        rowMeans = data.mean(axis=1, keepdims=True)
        colMeans = data.mean(axis=0, keepdims=True)
        matrixMean = data.mean()
        data = data - rowMeans - colMeans + matrixMean
        evals, evecs = LA.eigh(data)
        closeZero = np.isclose(evals, 0)
        evals[closeZero] = 0
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx][:, :dims_rescaled_data]
        evals = evals[idx][:dims_rescaled_data]
        return  (evecs * np.sqrt(evals)).tolist(), evals, evecs

    def NMMS(self, distance, normalize =1, alpha = 1, iteration = 50):
        xini = np.array( self.PCoA(distance)[0]) 
        n = len(xini)
        m = int(n * (n + 1) / 2 - n)
        matrix = [[0] * 8 for i in range(m)]
        k = 1
        i = 0
        for k in range(n):
            for j in range(k + 1, n):
                matrix[i][0] = k
                matrix[i][1] = j
                matrix[i][2] = distance[matrix[i][0]][matrix[i][1]]
                matrix[i][3] = np.linalg.norm(xini[matrix[i][0]] - xini[matrix[i][1]])
                i = i + 1
            k = k + 1
        matrix = np.array(matrix)
        matrix = matrix[matrix[:, 2].argsort()]
        sw = True
        it=0
        while (sw):
            i = 0
            while (i < m):
                temp = matrix[0][3]
                sum = matrix[i][3]
                count = 1
                pp = sum / count
                while (i < m - 1 and pp > matrix[i + 1][3]):
                    sum = sum + matrix[i + 1][3]
                    count = count + 1
                    pp = sum / count
                    i = i + 1
                while (count > 0):
                    matrix[i - count + 1][4] = pp
                    count = count - 1
                i = i + 1
            for i in range(m):
                matrix[i][5] = (matrix[i][3] - matrix[i][4]) ** 2
                matrix[i][6] = matrix[i][3] * matrix[i][3]
                matrix[i][7] = (matrix[i][3] - np.mean(matrix[:, 3])) ** 2
            if (np.sqrt((matrix[:, 5]).sum() / (matrix[:, 7]).sum()) < 0.001 or it > iteration):
                sw = False
            for i in range(n):
                sum = 0
                sum2 = 0
                for j in range(n):
                    if (i != j):
                        for k in range(m):
                            if (matrix[k][0] == i and matrix[k][1] == j) or (matrix[k][0] == j and matrix[k][1] == i):
                                sum = sum + (1 - matrix[k][4] / matrix[k][3]) * (xini[j][0] - xini[i][0])
                                sum2 = sum2 + (1 - matrix[k][4] / matrix[k][3]) * (xini[j][1] - xini[i][1])
                xini[i][0] = float(xini[i][0] + (alpha / (n - 1)) * sum)
                xini[i][1] = float(xini[i][1] + (alpha / (n - 1)) * sum2)
            for i in range(m):
                matrix[i][3] = np.linalg.norm(xini[int(matrix[i][0])] - xini[int(matrix[i][1])])
            print('iteration = '+ str(it))
            it=it+1
        if (normalize == 1):
            xini = xini / np.linalg.norm(xini)
        return xini


    def euclidianDistance(self, matrix):
        n = len(matrix[0])
        euclidean = [[0.0] * n for i in range(n)]
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                column1 = np.array([row[i] for row in matrix])
                column2 = np.array([row[j] for row in matrix])
                euclidean[i][j] = np.linalg.norm(column1 - column2)
                euclidean[j][i] = euclidean[i][j]
        return euclidean

    def permanova(self, matrix, grouping, permutations=999):
        distances = np.array(matrix)
        N = len(matrix)
        tri_idxs = np.triu_indices(N, k=1)
        distances = distances[tri_idxs]
        groups, grouping = np.unique(grouping, return_inverse=True)
        nn = len(groups)
        group_sizes = np.bincount(grouping)
        sT = (distances ** 2).sum()/ N
        Fi = np.empty(permutations, dtype=np.float64)
        F = 0
        for i in range(permutations+1):
            grouping_matrix = -1 * np.ones((N, N), dtype=int)
            for group_idx in range(nn):
                indices=np.where(grouping == group_idx)[0]
                within_indices = np.tile(indices, len(indices)), np.repeat(indices, len(indices))
                grouping_matrix[within_indices] = group_idx
            grouping_tri = grouping_matrix[tri_idxs]
            sW = 0
            for j in range(nn):
                sW += (distances[grouping_tri == j] ** 2).sum() / group_sizes[j]
            sA = sT - sW
            if (i == 0):
                F =  (sA / (nn - 1)) / (sW / (N - nn))
            else:
                Fi[i-1] = (sA / (nn - 1)) / (sW / (N - nn))
            grouping = np.random.permutation(grouping)
        P = ((Fi >= F).sum() + 1) / (permutations + 1)
        return P, Fi, F

    def testPer(self, dist, group):
        per=self.permanova(dist,group)
        print(per[0])
        print(per[2])
        print( permanova(DistanceMatrix(dist, range(len(group))), group,  column=None, permutations=999))

    def testPCoA(self, dist):
        sklearn_pcoa = Pcoa(dist)
        print('MYPCoA');
        print(self.PCoA(dist, 26)[0])
        print('PCoALibrary');
        print(sklearn_pcoa.samples)

    def testPCA(self, dist):
        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = -1*sklearn_pca.fit_transform(dist)
        print('MYPCA');
        print(self.PCA(dist, 2)[0])
        print('PCALibrary');
        print(sklearn_transf)

    def nms(self, data):
        xo=self.PCoA(data,2)[0];


    def plot(self, headers, points):
        group={}
        for i in range(len(points)):
            if not (headers[i] in group):
                group[headers[i]] = {}
                group[headers[i]]['x']=[]
                group[headers[i]]['y']=[]
            group[headers[i]]['x'].append(points[i][0])
            group[headers[i]]['y'].append(points[i][1])


        line1=[]
        labels=[]
        for key, value in group.items():
            temp, = pp.plot(value['x'],value['y'],'.');
            line1.append(temp)
            labels.append(key)
        pp.legend(line1,labels)
        pp.show()

