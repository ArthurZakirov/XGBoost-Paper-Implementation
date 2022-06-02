import numpy as np

class Leaf():
    """Base Class for Leafs/ Roots of all types
    
    Attributes
    ----------
    tree : Tree
        reference to the tree the leaf belongs to
        
    parent : Leaf/Root
        reference to the node one hierarchy level above
        
    tested : bool
        variable used during tree search that tells wether
        everything beneath that tree has been already checked
        True -> step to the other leaf of the parent node   
        
    x : np.array [N, D]
        input data
        
    r : np.array [N, 1]
        residuals
        
    p : np.array [N, 1]
        probability of output (only relevant for Classification)
        
    depth : int
        current level of inside the tree hierarchy
        
    Methods
    -------
    to_root() -> void
        Change Class of object from Leaf-child to Root-child
        
    can_be_split() -> bool
        Decide whether obj of class Leaf-child should be changed to Root-child
        
    leaf_metric() -> np.array []
    leaf_output() -> np.array []
    """
    def __init__(self, tree, parent, x, r, p, depth = 1):
        self.tree = tree
        self.parent = parent
        self.tested = False
        
        self.x = x
        self.r = r
        self.p = p
        
        self.depth = depth
        
    def to_root(self):
        """Change Class of object from Leaf-child to Root-child
        """
        if type(self) == self.tree.leaf_cls:
            self.__class__ = self.tree.root_cls
            self.__init__(self.tree, self.parent, self.x, self.r, self.p, self.depth)
        
    def can_be_split(self):
        """Decide whether obj of class Leaf-child should be changed to Root-child
        
        Arguments
        ---------
        cond_1 : bool
            make sure the leaf contains at least 2 samples, so that splitting even makes sense
            
        cond_2 : bool
            the tree has not yet grown to its final depth
        
        cond_3 : bool
            self is instance of a Leaf-child, not of Root-child
            That is important because during the init of Root this method will be used as well, 
            since Root is a child of Leaf.
            However we only want to test if a Leaf can be split, 
            otherwise there will be an infinite init loop. 
            
        Returns
        -------
        can_be_sṕlit : bool
        
        """
        cond_1 = (self.x.shape[0] >= 2)
        cond_2 = (self.depth <= self.tree.max_depth)
        
        can_be_sṕlit = cond_1 and cond_2
        return can_be_sṕlit 
    
    def leaf_metric(self):
        """Error or Simliarity metric of a leaf.
        """
        raise NotImplementedError()
        
    def leaf_output(self):
        """Formula that gives a single output for all residuals in a leaf.
        """
        raise NotImplementedError()
        
class XGBLeaf(Leaf):
    """Leaf with the specific XGBoost split criteria and output
    
    Attributes
    ----------
    lampda : np.array []
        regularization constant
        
    gamma : np.array []
        minimum gain value as pruning condition
        
    Returns
    -------
    leaf_metric() -> np.array [] 
    leaf_output() -> np.array []
    cover()       -> np.array []
    """
    def __init__(self, tree, parent, x, r, p, depth = 1):
        super(XGBLeaf, self).__init__(tree, parent, x, r, p, depth)
        self.lampda = self.tree.lampda
        self.gamma = self.tree.gamma
        
    def leaf_metric(self):
        """Similiarity of residuals inside the leaf
        Arguments
        ---------
        r : np.array [N,1]
        p : np.array [N,1]
        lampda : np.array []

        Returns
        -------
        similiarity : np.array []
        """
        similiarity = (self.r.sum()[np.newaxis] ** 2) / (self.cover() + self.lampda)
        return similiarity

    def leaf_output(self):
        """
        Arguments
        ---------
        r : np.array [N,1]
        p : np.array [N,1]
        lampda : np.array []

        Returns
        -------
        gamma_mj : np.array []
        """
        gamma_mj = self.r.sum()[np.newaxis] / (self.cover() + self.lampda)
        return gamma_mj
    
    def cover(self):
        """Hessian of the Loss function
        """
        raise NotImplementedError()
    
    
    
class XGBLeafRegression(XGBLeaf):
    def __init__(self, tree, parent, x, r, p, depth = 1):
        super(XGBLeafRegression, self).__init__(tree, parent, x, r, p, depth)
    
    def cover(self):
        """Hessian of the loss: number of residuals in leaf

        Arguments
        ---------
        r : np.array [N,1]

        Returns
        -------
        cover : np.array []

        """
        return len(self.r)

    
class XGBLeafClassification(XGBLeaf):
    def __init__(self, tree, parent, x, r, p, depth = 1):
        super(XGBLeafClassification, self).__init__(tree, parent, x, r, p, depth)
    
    def cover(self):
        """Hessian of the loss: Gini Index 

        Arguments
        ---------
        p : np.array [N,1]

        Returns
        -------
        cover : np.array []

        """
        q = np.ones_like(self.p) - self.p
        cover = (self.p * q).sum()
        return cover

     
def root_inherit_from(base):
    class Root(base):
        """Base Class for Decision Tree

        Attributes
        ----------
        d_split : np.array []
            dimension along which the split is performed
            tells which input feature should be compared to boundary

        x_d_split : np.array []
            x value of highest split gain along d_split
            tells the boundary to compare to during output assignment

        metric_opt : np.array []
            highest split gain
            tells wether split should be kept during pruning
            
        leaf_1 : Leaf/Root
        leaf_2 : Leaf/Root
        
        Methods
        -------
        find_best_spĺit()
            Get parameters for the optimal split into 2 leafs
            
        root_metric()
            Metric that decides how useful the split is
            
        get_boundaries()
            Return all boundaries that should be tested for a given dim
            
        split_root()
            Given the optimal split parameters, create 2 new leafs
            
        grow_tree_further()
            If possible, turn leafs of the root into new roots
        """
        def __init__(self, tree, parent, x, r, p, depth = 1):
            super(Root, self).__init__(tree, parent, x, r, p, depth)
            (self.d_split,
             self.x_d_split,
             self.metric_opt) = self.find_best_split(x, r, p)

            (self.leaf_1,
             self.leaf_2) = self.split_root(x, r, p, self.d_split, self.x_d_split) 

            self.grow_tree_further()



        ##############################################################################
        # TREE CREATION ##############################################################
        ##############################################################################

        def find_best_split(self, x, r, p):
            """Get parameters for the optimal split into 2 leafs
            
            Arguments
            ---------
            x : np.array [N, d]
            r : np.array [N, 1]
            p : np.array [N, 1]

            Returns
            -------
            d_split : np.array []
            x_d_split : np.array []
            metric_opt : np.array []
            """
            metric_opt = -np.inf

            # for every feature of the input data
            for d in range(x.shape[-1]):

                # sort samples so that the d'th feature increases in a monotonous way
                idx = x[:,d].argsort()
                x_d = x[idx, d]
                r = r[idx]
                p = p[idx]

                # for every possible boundary 
                splits = self.get_boundaries(x_d)
                for split in splits:

                    # create two new leafs
                    leaf_1, leaf_2 = self.split_root(x, r, p, d, split)

                    # check if metric val beats so far best metric val
                    metric = self.root_metric(leaf_1, leaf_2)

                    if metric > metric_opt:
                        metric_opt = metric
                        d_split = d
                        x_d_split = split

            return d_split, x_d_split, metric_opt


        def root_metric(self, leaf_1, leaf_2):
            """Similiarity gain

            Arguments
            ---------
            leaf_1
            leaf_2

            Returns
            -------
            gain : np.array []
            """        
            gain = leaf_1.leaf_metric() \
                 + leaf_2.leaf_metric() \
                 - self.leaf_metric()
            return gain


        def get_boundaries(self, x_d):
            """Return all boundaries that should be tested for a given dim

            Arguments
            ---------
            x_d : np.array [N]

            Returns
            ------
            boundaries : list of np.array [0]
            """
            boundaries = [(x_d[i] + x_d[i+1])/2 for i in range(x_d.shape[0]-1)]
            return boundaries


        def split_root(self, x, r, p, d_split, x_d_split):
            """Given the optimal split parameters, create 2 new leafs

            Arguments
            ---------
            x : np.array [N, d]
            r : np.array [N, 1]
            d_split: int
            x_d_split : np.array []

            Returns
            -------
            leaf_1 : object of type Leaf-child
            leaf_2 : object of type Leaf-child
            """
            mask = x[:,d_split] < x_d_split
            leaf_1 = self.tree.leaf_cls(self.tree, self, x[mask], r[mask], p[mask], self.depth + 1)
            leaf_2 = self.tree.leaf_cls(self.tree, self, x[~mask], r[~mask], p[~mask], self.depth + 1)
            return leaf_1, leaf_2


        def grow_tree_further(self):
            """If possible, turn leafs of the root into new roots
            """
            if self.leaf_1.can_be_split():
                self.leaf_1.to_root()

            if self.leaf_2.can_be_split():
                self.leaf_2.to_root()


        ##########################################################################
        # PRUNING ################################################################
        ##########################################################################

        def prune_bool(self):
            """Tells wether root should be pruned

            Arguments
            ---------
            cond_1 : bool
                the error improvement from splitting is sufficient

            cond_2 : bool
                make sure to not delete entire tree

            Returns
            -------
            prune : bool

            """
            cond_1 = self.metric_opt < self.tree.gamma
            cond_2 = self.parent != self.tree
            prune = cond_1 and cond_2
            return prune

        def to_leaf(self):
            """Change Class of object from Root-child to Leaf-child
            """
            if type(self) == self.tree.root_cls:
                self.__class__ = self.tree.leaf_cls
                self.__init__(self.tree, self.parent, self.x, self.r, self.p, self.depth)
                
    return Root