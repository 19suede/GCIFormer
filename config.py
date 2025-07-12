def select_config(dataset):
    dataset_configs = {
        "QuickBird": QB_config,
        "WorldView3": WV3_config,
        "GaoFen2": GF2_config,
    }

    config = dataset_configs.get(dataset)

    if config:
        return config()
    else:
        raise ValueError(f"Dataset '{dataset}' config not found.")

class Config(object):
    def __init__(self):
        """
        Config for model on different dataset
        """
        self.patch_size = 4
        self.window_size = 8

class QB_config(Config):
    def __init__(self):
        super(QB_config, self).__init__()
        self.trainset_path = "/root/dataset/pansharpening/PanCollection/training_data/train_qb.h5"
        self.validset_path = "/root/dataset/pansharpening/PanCollection/validation_data/valid_qb.h5"
        self.testset_path_f = "/root/dataset/pansharpening/PanCollection/test_data/test_qb_OrigScale_multiExm1.h5"
        self.testset_path_r = "/root/dataset/pansharpening/PanCollection/test_data/test_qb_multiExm1.h5"

        self.pan_ch = 1 # pan channels
        self.lms_ch = 4 # lms channels
        self.dr = 2047.0
        self.num_samples = 17139

        self.patch_size = 4
        self.window_size = 8

        self.num_blocks = 3
        self.model_dim = 32
        self.hidden_ch = 48

class WV3_config(Config):
    def __init__(self):
        super(WV3_config, self).__init__()
        self.trainset_path = "/root/dataset/pansharpening/PanCollection/training_data/train_wv3.h5"
        self.validset_path = "/root/dataset/pansharpening/PanCollection/validation_data/valid_wv3.h5"
        self.testset_path_f = "/root/dataset/pansharpening/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5"
        self.testset_path_r = "/root/dataset/pansharpening/PanCollection/test_data/test_wv3_multiExm1.h5"

        self.pan_ch = 1  # pan channels
        self.lms_ch = 8  # lms channels
        self.dr = 2047.0
        self.num_samples = 9714

        self.patch_size = 4
        self.window_size = 8

        self.num_blocks = 4
        self.model_dim = 32
        self.hidden_ch = 64


class GF2_config(Config):
    def __init__(self):
        super(GF2_config, self).__init__()
        self.trainset_path = "/root/dataset/pansharpening/PanCollection/training_data/train_gf2.h5"
        self.validset_path = "/root/dataset/pansharpening/PanCollection/validation_data/valid_gf2.h5"
        self.testset_path_f = "/root/dataset/pansharpening/PanCollection/test_data/test_gf2_OrigScale_multiExm1.h5"
        self.testset_path_r = "/root/dataset/pansharpening/PanCollection/test_data/test_gf2_multiExm1.h5"

        self.pan_ch = 1  # pan channels
        self.lms_ch = 4  # lms channels
        self.dr = 1023.0
        self.num_samples = 19809

        self.patch_size = 4
        self.window_size = 8

        self.num_blocks = 4
        self.model_dim = 32
        self.hidden_ch = 48


if __name__=="__main__":
    config = select_config("WorldView3")
    print(config.model_dim)
