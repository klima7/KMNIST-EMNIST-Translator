from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture


class EnhancedSpectralClustering(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_clusters):
        self.clusterizer = SpectralClustering(n_clusters)
        self.classifier = GaussianMixture(n_clusters)
    
    def fit(self, X):            
        labels = self.clusterizer.fit_predict(X)
        self.classifier.fit(X, labels)
        
    def predict(self, X):
        return self.classifier.predict(X)
