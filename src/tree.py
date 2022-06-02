


    


class Tree():
    """
    Attributes
    ----------
    max_depth
        number of splits, counted along the vertical axis
        
    root_cls
        Class of root objects that will build the tree
        
    lef_cls
        Class of the Leaf objects that will build the tree
        
    root
        First split of the tree. 
        All roots and leafs of the tree are recursively contained inside this root.
        
    lampda : np.array []
        regularization constant
        
    gamma : np.array []
        minimum gain value as pruning condition
        
    Methods
    -------
    fit_tree()
        Fit Tree to the data
        
    prune_tree()
        From bottom up unite all not useful splits into a leaf
        
    single_sample_output()
        Go down the whole tree based on a single x_i sample
        
    output()
        For data x create predictions gamma
        
    search_tree()
        Walk trough all roots of the tree
        
    root_is_final_root()
        Set the attribute "tested" of all roots in the tree to False
        Must be done after searching / pruning
        
    step_from_intermediate_root()
        Get next root during tree search, 
        if root has root/root, root/leaf, leaf/root
        
    step_from_final_root()
        Get next root during tree search, if root has leaf leaf. 
        Prune if necessary.
        
    step_root()
        Recursively Step trough all roots of the tree, used for pruning
        
    """
    def __init__(self, LeafClass, RootClass, max_depth, lampda, gamma):
        self.max_depth = max_depth
        self.root_cls = RootClass
        self.leaf_cls = LeafClass
        
        self.root = None
        self.lampda = lampda
        self.gamma = gamma
        
    def fit_tree(self, x, r, p):
        """Fit Tree to the data
        """
        self.root = self.root_cls(tree=self, 
                                  parent=self,
                                  x=x,
                                  r=r,
                                  p=p)
        
    def prune_tree(self):
        """From bottom up unite all not useful splits into a leaf
        """
        self.step_root(self.root, prune_during_search = False)
        self.reset_search_attribute(self.root)
       

    def single_sample_output(self, root, x_i):
        """Go down the whole tree based on a single x_i sample
        Arguments
        ---------
        x_i : np.array [N]
        root
        
        Returns
        -------
        root
        """
        if type(root) == self.root_cls:
            if x_i[root.d_split] < root.x_d_split:
                return self.single_sample_output(root.leaf_1, x_i)
            else:
                return self.single_sample_output(root.leaf_2, x_i)
                
        if type(root) == self.leaf_cls:
            return root.leaf_output()
        
    def output(self, x):
        """For data x create predictions gamma
        
        Arguments
        ---------
        x : np.array [N, d]
        
        Returns
        -------
        gamma : np.array [N, 1]
        """
            
        r = np.stack([self.single_sample_output(self.root, x_i) for x_i in x])
        return r

        
    def search_tree(self):
        """Walk trough all roots of the tree
        """
        self.step_root(self.root, prune_during_search = False)
        self.reset_search_attribute(self.root)
        
    def reset_search_attribute(self, root):
        """Set the attribute "tested" of all roots in the tree to False. Must be done after searching / pruning.
        """
        if type(root) == self.root_cls:
            root.tested = False
            return self.reset_search_attribute(root.leaf_1), self.reset_search_attribute(root.leaf_2)
        
        
        
    def step_from_intermediate_root(self, root):
        """Get next root during tree search, if root has root/root, root/leaf, leaf/root
        
        Arguments
        ---------
        root
        
        Returns
        -------
        root
        """
        if not root.leaf_1.tested and not root.leaf_2.tested:
            return root.leaf_1
        if root.leaf_1.tested and not root.leaf_2.tested:
            return root.leaf_2
        if root.leaf_1.tested and root.leaf_2.tested:
            root.tested = True
            return root.parent
        
    def step_from_final_root(self, root, prune_during_search):
        """Get next root during tree search, if root has leaf leaf. Prune if necessary.
        
        Arguments
        ---------
        root
        
        Returns
        -------
        root
        """
        if root.prune_bool() and prune_during_search:
            root.to_leaf()
        root.tested = True
        return root.parent
    
    def root_is_final_root(self, root):
        """Determine wether root has two leafs
        
        Arguments
        ---------
        root : self.root_cls object
        
        Return
        ------
        root_has_leaf_leaf : bool
        """
        root_has_leaf_leaf = type(root.leaf_1) == self.leaf_cls and type(root.leaf_2) == self.leaf_cls
        return root_has_leaf_leaf
        
        
    def step_root(self, root, prune_during_search = False):
        """Recursively Step trough all roots of the tree, used for pruning
        """
        if type(root) == self.root_cls:
            if not self.root_is_final_root(root) :
                root = self.step_from_intermediate_root(root)
            else:
                root =  self.step_from_final_root(root, prune_during_search)
            return self.step_root(root)