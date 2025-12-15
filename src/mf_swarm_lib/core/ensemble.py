from os import path, mkdir
import json
from glob import glob

import numpy as np
from mf_swarm_lib.core.ml.multi_input_clf import MultiInputNet

# ensemble of models created using cross-validation, acting as one unit
# the results are the average of the results of the individual models
# this is used to reduce overfitting and improve generalization
# it is a simple ensemble that averages the predictions of the individual models
'''
4. Alternatives and Improvements
Weighted Averaging: Instead of a simple mean, use performance metrics (e.g., fold-specific accuracy) 
to weight predictions 10.
Stacking: Train a meta-model (e.g., logistic regression) on the CV models' predictions 
for smarter aggregation 1012.
Bootstrap Aggregating (Bagging): Combine CV with resampling to further reduce variance 28.
'''
class BasicEnsemble():
    def __init__(self, model_list, stats_dicts) -> None:
        models_and_fmax = [(m, s) for m, s in zip(model_list, stats_dicts)]
        models_and_fmax.sort(key=lambda x: x[1]['Fmax'])

        self.models = [m for m, s in models_and_fmax]
        self.stats_dicts = [s for m, s in models_and_fmax]
        self.stats = {}
        for k in self.stats_dicts[0].keys():
            self.stats[k] = round(np.mean([d[k] for d in self.stats_dicts]), 5)

    def predict(self, x, verbose=0, weights=[1, 2, 2, 2, 3]):
        results = [m.predict(x) for m in self.models]
        results_weighted = np.average(results, axis=0)
        return results_weighted

    def save(self, output_dir):
        if not path.exists(output_dir):
            mkdir(output_dir)
        paths = [output_dir + '/fold_'+str(n)
            for n in range(len(self.models))]
        for m, p in zip(self.models, paths):
            m.save(p)
        json.dump(self.stats_dicts, open(output_dir+'/stats_dicts.json', 'w'), indent=4)
    
    def load(models_dir):
        fold_dirs = glob(models_dir+'/fold_*')
        fold_dirs.sort(key = lambda p: int(p.split('_')[-1]))
        print('Base models found at {models_dir}/fold_*: ', fold_dirs)
        model_list = [MultiInputNet.load(d) for d in fold_dirs]
        stats_dicts = json.load(open(models_dir+'/stats_dicts.json', 'r'))

        return BasicEnsemble(model_list, stats_dicts)
