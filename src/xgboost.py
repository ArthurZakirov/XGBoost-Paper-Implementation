import numpy as np
import scipy.special as ss

class XGBoost():
    """
    Attributes
    ----------
    num_trees : int
        corresponds to M. There are m = 1,...,M trees. 
        However there are M+1 models, since for m=0 there is a constant F_0.
        
    learning_rate : np.array []
        Stepsize of the Gradient Descent
    
    F_0 : np.array [N, 1]
        Initial Prediction, usually the mean of the labels.
    
    root_cls
        Class of root objects that will build the trees of XGBoost
        
    tree : list of Tree
    
    Methods
    -------
    loss()
    residuals()
    initial_prediction()
    output()
    fit()
    """
    def __init__(self, num_trees, max_depth, learning_rate, lampda, gamma):
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        
        self.F_0 = 0.
        
        self.root_cls = root_inherit_from(self.leaf_cls)
        self.trees = [Tree(self.leaf_cls, 
                           self.root_cls, 
                           max_depth,
                           lampda,
                           gamma) 
                      for m in range(num_trees)] 
        
    def loss_fn(self):
        """
        Arguments
        ---------
        F_m : np.array [N, 1]  
        p_m : np.array [N, 1]
        y : np.array [N, 1]
            
        Returns
        -------
        L_m : np.array []
        """
        raise NotImplementedError()
    
    
    def residual_fn(self):
        """
        Arguments
        ---------
        F_m : np.array [N, 1]            
        y : np.array [N, 1]
            
        Returns
        -------
        r_m : np.array [N, 1]
        """
        raise NotImplementedError()
    
    
    def initial_prediction(self):
        """Prediction of first model (not a tree)
        
        Arguments
        ---------
        y : np.array [N, 1]
        
        Returns
        -------
        F_m : np.array [N, 1]
        p_m : np.array [N, 1]
        """
        raise NotImplementedError()
        
        
    def output_layer(self):
        """Apply final layer to convert Model output to labels
        
        Arguments
        ---------
        F_m : np.array [N, 1]
        
        Returns
        -------
        y_pred : np.array [N, 1]
        """
        raise NotImplementedError()
    
    
    def fit(self, x, y):
        """Fit Model to the data
        
        Arguments
        ---------
        x : np.array [N, d]
            input data
            
        y : np.array [N, 1]
            labels
            
        Returns
        -------
        None
        """
        loss = list()
        F_m, p_m = self.initial_prediction(y)
        for m in range(self.num_trees):
            # Loss
            L_m = self.loss_fn(y, F_m, p_m)
            loss.append(L_m)
            
            # Fit tree
            r_m = self.residual_fn(y, F_m)
            self.trees[m].fit_tree(x, r_m, p_m)
            
            # Update Model output
            gamma_m = self.trees[m].output(x)
            F_m = F_m + self.learning_rate * gamma_m
            p_m = ss.expit(F_m)
        
        self.loss = loss
        
        
    def predict(self, x):
        """Predict new labels for unseen input data
        
        Arguments
        ---------
        x : np.array [N, d]
        
        Returns
        -------
        y_pred : np.array [N, 1]
        """
        F_m = self.F_0
        for m in range(self.num_trees):
            gamma_m = self.trees[m].output(x)
            F_m = F_m + self.learning_rate * gamma_m
        y_pred = self.output_layer(F_m)
        return y_pred
    
    
class XGBoostRegression(XGBoost):
    def __init__(self, num_trees, max_depth, learning_rate, lampda, gamma):
        self.leaf_cls = XGBLeafRegression
        super(XGBoostRegression, self).__init__(num_trees, max_depth, learning_rate, lampda, gamma) 
            
            
    def loss_fn(self, y, F_m, p_m):
        L_m = ((y - F_m) ** 2).mean()
        return L_m
    
    
    def residual_fn(self, y, F_m):
        r_m = y - F_m   
        return r_m
    
    
    def initial_prediction(self, y):
        F_m = y.mean() * np.ones_like(y)
        p_m = F_m
        self.F_0 = F_m
        return F_m, p_m
    
    
    def output_layer(self, F_m):
        return F_m
    
class XGBoostClassification(XGBoost):
    def __init__(self, num_trees, max_depth, learning_rate, lampda, gamma):
        self.leaf_cls = XGBLeafClassification
        super(XGBoostClassification, self).__init__(num_trees, max_depth, learning_rate, lampda, gamma) 
    
    
    def loss_fn(self, y, F_m, p_m):
        q_m = np.ones_like(p_m) - p_m
        L_m = - (p_m * np.log(p_m) + q_m * np.log(q_m)).sum()
        return L_m
    
    
    def residual_fn(self, y, F_m):
        r_m = y - ss.expit(F_m)  
        return r_m
    
    
    def initial_prediction(self, y):
        p_m = y.mean() * np.ones_like(y)
        F_m = ss.logit(p_m)
        self.F_0 = F_m
        return F_m, p_m

    
    def output_layer(self, F_m):
        return ss.expit(F_m)