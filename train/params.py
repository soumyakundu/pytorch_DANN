from models import models

# utility params
fig_mode = None
embed_plot_epoch=10

# model params
use_gpu = True
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

batch_size = 512
epochs = 1000
gamma = 10
theta = 1

# path params
data_root = './data'

mnist_path = data_root + '/MNIST'
mnistm_path = data_root + '/MNIST_M'
svhn_path = data_root + '/SVHN'
syndig_path = data_root + '/SynthDigits'

save_dir = './experiment'


# specific dataset params
extractor_dict = {'MNIST_MNIST_M': models.Img_Extractor(),
                  'SVHN_MNIST': models.SVHN_Extractor(),
                  'Human_Mouse': models.Extractor(),
                  'SynDig_SVHN': models.SVHN_Extractor()}

class_dict = {'MNIST_MNIST_M': models.Img_Class_classifier(),
              'SVHN_MNIST': models.SVHN_Class_classifier(),
              'Human_Mouse': models.Class_classifier(),
              'SynDig_SVHN': models.SVHN_Class_classifier()}

domain_dict = {'MNIST_MNIST_M': models.Img_Domain_classifier(),
               'SVHN_MNIST': models.SVHN_Domain_classifier(),
               'Human_Mouse': models.Domain_classifier(),
               'SynDig_SVHN': models.SVHN_Domain_classifier()}
