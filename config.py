import yaml

class Modelcfg():
    def __init__(self, cfg):
        self.input_shape = cfg['input_shape']
        self.num_classes = cfg['num_classes']
        if cfg['type'] == "B0":
            #Overlap patch merging module params
            self.kernel_sizes=[7, 3, 3, 3]
            self.strides=[4, 2, 2, 2]
            self.emb_sizes=[32, 64, 160, 256]
            #Reduction
            self.reduction_ratios=[8, 4, 2, 1]
            self.mlp_expansions=[8, 8, 4, 4]
            #Cross attention params
            self.num_heads=[1, 2, 5, 8]
            self.depths=[2, 2, 2, 2]
            #decoder params
            self.decoder_channels = 256
            self.scale_factors = [1, 2, 4, 8]
        if cfg['type'] == "B1":
            #Overlap patch merging module params
            self.kernel_sizes=[7, 3, 3, 3]
            self.strides=[4, 2, 2, 2]
            self.emb_sizes=[32, 64, 160, 256]
            #Reduction
            self.reduction_ratios=[8, 4, 2, 1]
            self.mlp_expansions=[8, 8, 4, 4]
            #Cross attention params
            self.num_heads=[1, 2, 5, 8]
            self.depths=[2, 2, 2, 2]
            #decoder params
            self.decoder_channels = 256
            self.scale_factors = [1, 2, 4, 8]
        if cfg['type'] == "B2":
            #Overlap patch merging module params
            self.kernel_sizes=[7, 3, 3, 3]
            self.strides=[4, 2, 2, 2]
            self.emb_sizes=[32, 64, 160, 256]
            #Reduction
            self.reduction_ratios=[8, 4, 2, 1]
            self.mlp_expansions=[8, 8, 4, 4]
            #Cross attention params
            self.num_heads=[1, 2, 5, 8]
            self.depths=[2, 2, 2, 2]
            #decoder params
            self.decoder_channels = 256
            self.scale_factors = [1, 2, 4, 8]
        if cfg['type'] == "B3":
            #Overlap patch merging module params
            self.kernel_sizes=[7, 3, 3, 3]
            self.strides=[4, 2, 2, 2]
            self.emb_sizes=[32, 64, 160, 256]
            #Reduction
            self.reduction_ratios=[8, 4, 2, 1]
            self.mlp_expansions=[8, 8, 4, 4]
            #Cross attention params
            self.num_heads=[1, 2, 5, 8]
            self.depths=[2, 2, 2, 2]
            #decoder params
            self.decoder_channels = 256
            self.scale_factors = [1, 2, 4, 8]
        if cfg['type'] == "B4":
            #Overlap patch merging module params
            self.kernel_sizes=[7, 3, 3, 3]
            self.strides=[4, 2, 2, 2]
            self.emb_sizes=[32, 64, 160, 256]
            #Reduction
            self.reduction_ratios=[8, 4, 2, 1]
            self.mlp_expansions=[8, 8, 4, 4]
            #Cross attention params
            self.num_heads=[1, 2, 5, 8]
            self.depths=[2, 2, 2, 2]
            #decoder params
            self.decoder_channels = 256
            self.scale_factors = [1, 2, 4, 8]
        if cfg['type'] == "B5":
            #Overlap patch merging module params
            self.kernel_sizes=[7, 3, 3, 3]
            self.strides=[4, 2, 2, 2]
            self.emb_sizes=[32, 64, 160, 256]
            #Reduction
            self.reduction_ratios=[8, 4, 2, 1]
            self.mlp_expansions=[8, 8, 4, 4]
            #Cross attention params
            self.num_heads=[1, 2, 5, 8]
            self.depths=[2, 2, 2, 2]
            #decoder params
            self.decoder_channels = 256
            self.scale_factors = [1, 2, 4, 8]

class Datacfg():
    def __init__(self, cfg):
        self.train_image_dir = cfg['train_image_dir']
        self.train_mask_dir = cfg['train_mask_dir']
        self.val_image_dir = cfg['val_image_dir']
        self.val_mask_dir = cfg['val_mask_dir']

class Trainingcfg():
    def __init__(self, cfg):
        self.batch_size = cfg['batch_size']
        self.num_epochs = cfg['num_epochs']
        self.resume_training = cfg['resume_training']
        self.weights_path = cfg['weights_path']

class Config():
    def __init__(self, config_path="config.yml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        try:
            self.model = Modelcfg(self.cfg['model'])
            self.data = Datacfg(self.cfg['data'])
            self.training = Trainingcfg(self.cfg['training'])
        except Exception as e:
            print("Error loading config file...")
            print(str(e))
