

import matplotlib.pyplot as plt
import seaborn as sns

def plot_tree(dtmodel, feature_names, file_name='tree.png'):
    from sklearn import tree
    plt.figure(figsize=(20, 10))
    tree.plot_tree(dtmodel, feature_names=feature_names, filled=True)
    plt.savefig(file_name, dpi=300)
    plt.show()