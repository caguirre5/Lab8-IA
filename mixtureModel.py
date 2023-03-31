import numpy as np
import pandas as pd


class GMM:
    '''
        Esta clase implementa el modelo Gausiano Mixture Models
        basado en la implementación de la libería sci-kit learn.
    '''

    def __init__(self, n_components, max_iter=20, comp_names=None):
        '''
            Esta función inicializa el modelo configurando los siguientes parámetros:
                :param n_components: int
                    El número de clusters en los que el algoritmo debe dividir el conjunto de datos
                :param max_iter: int, default = 100
                    El número de iteraciones que el algoritmo realizará para encontrar los clusters
                :param comp_names: lista de strings, default=None
                    En caso de que se establezca como una lista de cadenas se utilizará para
                    nombrar los clusters.
        '''
        self.n_componets = n_components
        self.max_iter = max_iter
        if comp_names == None:
            self.comp_names = [
                f"comp{index}" for index in range(self.n_componets)]
        else:
            self.comp_names = comp_names
        # pi lista contiene la fracción del conjunto de datos para cada cluster
        self.pi = [1/self.n_componets for comp in range(self.n_componets)]

    def multivariate_normal(self, X, mean_vector, covariance_matrix):
        '''
            Esta función implementa la fórmula de derivación normal multivariante,
            la distribución normal para vectores requiere los siguientes parámetros:
                :param X: numpy array de 1 dimensión
                    El vector fila para el cual deseamos calcular la distribución.
                :param mean_vector: numpy array de 1 dimensión
                    El vector fila que contiene las medias para cada columna.
                :param covariance_matrix: numpy array de 2 dimensiones (matriz)
                    La matriz 2-d que contiene las covarianzas de las características.
        '''
        return (2*np.pi)**(-len(X)/2)*np.linalg.det(covariance_matrix)**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)

    def fit(self, X):
        '''
            Esta función es utilizada para entrenar el modelo.
                :param X: numpy array de 2 dimensiones
                    Los datos deben pasarse al algoritmo como una matriz 2-d,
                    donde las columnas son las características y las filas son las muestras.
        '''
        # División de los datos en n_componets subconjuntos
        new_X = np.array_split(X, self.n_componets)
        # Cálculo inicial del vector medio y la matriz de covarianza
        self.mean_vector = [np.mean(x, axis=0) for x in new_X]
        self.covariance_matrixes = [np.cov(x.T) for x in new_X]
        # Eliminando la matriz new_X porque no la necesitaremos más
        del new_X
        for iteration in range(self.max_iter):
            ''' --------------------------   E - STEP   -------------------------- '''
            # Iniciando la matriz r, cada fila contiene las probabilidades
            # para cada cluster para esta fila.
            self.r = np.zeros((len(X), self.n_componets))
            # Calculando la matriz r
            for n in range(len(X)):
                for k in range(self.n_componets):
                    self.r[n][k] = self.pi[k] * self.multivariate_normal(
                        X[n], self.mean_vector[k], self.covariance_matrixes[k])
                    self.r[n][k] /= sum([self.pi[j]*self.multivariate_normal(
                        X[n], self.mean_vector[j], self.covariance_matrixes[j]) for j in range(self.n_componets)])
            # Se calcula N
            N = np.sum(self.r, axis=0)
            ''' --------------------------   M - STEP   -------------------------- '''
            # Inicializar el vector de medias como un vector de ceros
            self.mean_vector = np.zeros((self.n_componets, len(X[0])))
            # Actualizar el vector de medias
            for k in range(self.n_componets):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
            self.mean_vector = [1/N[k]*self.mean_vector[k]
                                for k in range(self.n_componets)]
            # Inicializar la lista de matrices de covarianza
            self.covariance_matrixes = [
                np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_componets)]
            # Actualizar las matrices de covarianza
            for k in range(self.n_componets):
                self.covariance_matrixes[k] = np.cov(
                    X.T, aweights=(self.r[:, k]), ddof=0)
            self.covariance_matrixes = [
                1/N[k]*self.covariance_matrixes[k] for k in range(self.n_componets)]
            # Actualizar la lista pi
            self.pi = [N[k]/len(X) for k in range(self.n_componets)]

    def predict(self, X):
        '''
            Función para predecir la pertenencia a los clusters.
                :param X: numpy array 2D
                    Los datos en los que debemos predecir los clusters.
        '''
        # Inicializamos la matriz de probabilidades
        probabilities = []
        # Calculamos las probabilidades para cada cluster
        for n in range(len(X)):
            probabilities.append([self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                                  for k in range(self.n_componets)])
        cluster = []
        # Devolvemos los nombres de los clusters correspondientes a los índices
        for probabilitie in probabilities:
            cluster.append(probabilitie.index(max(probabilitie)) + 1)
        return cluster
