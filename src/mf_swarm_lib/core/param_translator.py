network_mandatory_params = ['l1_dim', 'l2_dim', 'dropout_rate', 'leakyrelu_1_alpha']

param_types = {
    'l1_dim': int,
    'l2_dim': int,
    'leakyrelu_1_alpha': float,
    'dropout_rate': float,
    'final_dim': int,
    'patience': int,
    'epochs': int,
    'learning_rate': float,
    'batch_size': int
}

class ProblemTranslator:
    def __init__(self, my_param_bounds, raw_values= None) -> None:
        if raw_values != None:
            self.params_list = raw_values['params_list']
            self.upper_bounds = raw_values['upper_bounds']
            self.lower_bounds = raw_values['lower_bounds']
        else:
            self.params_list = []
            self.upper_bounds = []
            self.lower_bounds = []
            param_groups = sorted(my_param_bounds.keys())
            for key in param_groups:
                param_names = sorted(my_param_bounds[key].keys())
                for name in param_names:
                    lower = float(my_param_bounds[key][name][0])
                    upper = float(my_param_bounds[key][name][1])
                    self.lower_bounds.append(lower)
                    self.upper_bounds.append(upper)
                    self.params_list.append((key, name))
        
        '''print('Param bounds:')
        for i in range(len(self.params_list)):
            print(self.params_list[i], self.upper_bounds[i], self.lower_bounds[i])'''

    '''def to_bounds(self):
        return FloatVar(lb=self.lower_bounds, ub=self.upper_bounds)'''
    
    def to_dict(self):
        return {
            'params_list': self.params_list,
            'upper_bounds': self.upper_bounds,
            'lower_bounds': self.lower_bounds,
        }

    def decode(self, vec):
        new_param_dict = {}
        for first, second in self.params_list:
            new_param_dict[first] = {}
        for i in range(len(self.params_list)):
            first, second = self.params_list[i]
            converter = param_types[second]
            val = converter(vec[i])
            new_param_dict[first][second] = val
        
        for param_group_name, params in new_param_dict.items():
            if param_group_name.startswith('esm') or param_group_name.startswith('taxa'):
                for param_name in network_mandatory_params:
                    if not param_name in params:
                        print(new_param_dict)
                        print('Cannot find', param_name, 'in', params)
                        raise Exception("Could not find " + param_name)
        
        return new_param_dict

    def encode(self, param_dict):
        vec = []
        for i in range(len(self.params_list)):
            first, second = self.params_list[i]
            original = param_dict[first][second]
            val = float(original)
            vec.append(val)

        #print('Encoded solution', param_dict, 'to')
        #print(vec)
        return vec