from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster as sc
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralBiclustering
from memory_profiler import profile
#@profile
def k_means_model(total_random_generate,n_clusters = 4):
    X = [inst['output'] for inst in total_random_generate]

    y_pred = KMeans(n_clusters=n_clusters, random_state=9,n_init=10).fit_predict(X)
    rs_index_dict = {}
    for i in range(n_clusters):
        rs_index_dict[i] = []
    for idx in range(len(total_random_generate)):
        rs_index_dict[y_pred[idx]].append(total_random_generate[idx]['output'])
        total_random_generate[idx]['y_class'] = y_pred[idx]
    return rs_index_dict,total_random_generate
#@profile
def affinity(total_random_generate,n_clusters = 4):
    X = [inst['output'] for inst in total_random_generate]

    model = AffinityPropagation(damping=0.5, max_iter=500, convergence_iter=30,preference=-50).fit(X)
    y_pred = model.labels_
    rs_index_dict = {}
    for i in range(n_clusters):
        rs_index_dict[i] = []
    for idx in range(len(total_random_generate)):
        rs_index_dict[y_pred[idx]].append(total_random_generate[idx]['output'])
        total_random_generate[idx]['y_class'] = y_pred[idx]
    return rs_index_dict, total_random_generate
#@profile
def sc_Clustering(total_random_generate,n_clusters):
    X = [inst['output'] for inst in total_random_generate]
    y_pred = sc.SpectralClustering(gamma=1, n_clusters=n_clusters,affinity='nearest_neighbors').fit_predict(X)
    rs_index_dict = {}
    for i in range(n_clusters):
        rs_index_dict[i] = []
    for idx in range(len(total_random_generate)):
        rs_index_dict[y_pred[idx]].append(total_random_generate[idx]['output'])
        total_random_generate[idx]['y_class'] = y_pred[idx]
    return rs_index_dict, total_random_generate
#@profile
def Agglomerative(total_random_generate,n_clusters):
    X = [inst['output'] for inst in total_random_generate]

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    y_pred = model.fit(X).labels_
    rs_index_dict = {}
    for i in range(n_clusters):
        rs_index_dict[i] = []
    for idx in range(len(total_random_generate)):
        rs_index_dict[y_pred[idx]].append(total_random_generate[idx]['output'])
        total_random_generate[idx]['y_class'] = y_pred[idx]
    return rs_index_dict, total_random_generate
#@profile
def birch(total_random_generate,n_clusters):
    X = [inst['output'] for inst in total_random_generate]

    model = Birch(n_clusters = n_clusters)
    y_pred = model.fit_predict(X)
    rs_index_dict = {}
    for i in range(n_clusters):
        rs_index_dict[i] = []
    for idx in range(len(total_random_generate)):
        rs_index_dict[y_pred[idx]].append(total_random_generate[idx]['output'])
        total_random_generate[idx]['y_class'] = y_pred[idx]
    return rs_index_dict, total_random_generate
#@profile
def Gauss(total_random_generate,n_clusters):
    X = [inst['output'] for inst in total_random_generate]
    model = GaussianMixture(n_components=n_clusters)
    y_pred = model.fit_predict(X)
    rs_index_dict = {}
    for i in range(n_clusters):
        rs_index_dict[i] = []
    for idx in range(len(total_random_generate)):
        rs_index_dict[y_pred[idx]].append(total_random_generate[idx]['output'])
        total_random_generate[idx]['y_class'] = y_pred[idx]
    return rs_index_dict, total_random_generate
#@profile
def Spectral(total_random_generate,n_clusters):
    X = [inst['output'] for inst in total_random_generate]
    model = SpectralBiclustering(n_clusters=n_clusters)
    model.fit(X)
    y_pred = model.row_labels_
    rs_index_dict = {}
    for i in range(n_clusters):
        rs_index_dict[i] = []
    for idx in range(len(total_random_generate)):
        rs_index_dict[y_pred[idx]].append(total_random_generate[idx]['output'])
        total_random_generate[idx]['y_class'] = y_pred[idx]
    return rs_index_dict, total_random_generate