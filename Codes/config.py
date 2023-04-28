# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

# TensorFlow options.
tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.
tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']          = True     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
#env.CUDA_VISIBLE_DEVICES                       = '0'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '0'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.
#----------------------------------------------------------------------------
desc        = 'pgan3D'                                        # Description string included in result subdir name.
random_seed = 8001                                          # Global random seed.
dataset     = EasyDict(tfrecord_dir='TrainingData')         # Options for dataset.load_dataset(). dataset is from 'TrainingData' folder of data_dir 
train       = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')      # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss.D_wgangp_acgan')          # Options for discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
#----------------------------------------------------------------------------

###########################
# Set the following parameters according to user-defined applications.
###########################

#----------------------------------------------------------------------------
# Uncomment the following lines to further train the pre-trained GANs
# train.resume_run_id = '/scratch/users/suihong/SubwaterFan/TrainedModels/015-pgan3D128x128x32_4gpu_softmax/network-snapshot-011525.pkl'                                                     # Run ID or network pkl to resume training from, None = start from scratch.
# train.resume_kimg = 11525  # Assumed training progress at the beginning. Affects reporting and training schedule.
# train.resume_time = 48*3600 # seconds, Assumed wallclock time at the beginning. Affects reporting.
#----------------------------------------------------------------------------
desc += '128x128x32_4gpu_softmax';   # Supplement descriptions onto the folder name of results
num_gpus = 4   # number of gpus used for training
train.total_kimg = 900000   # thousands of training data used before stopping
G.facies_codes = [0, 1, 2]    # facies code value in training dataset
G.beta = 8e3  # Used in soft-argmax function, to be tuned for specific cases; 8e3 works ok for my case where we have 3 facies: channel, lobe, and mud. If the final trained generator produces isolated facies volumes, especially isolated single-pixel facies around the surface of another facies, try to increase beta several times; G is very sensitive to beta, so beta should not be set too large in which case vanishing gradient may occur (I haven't tested).

G.num_facies = len(G.facies_codes)            # Number of facies types.
dataset.data_range = [min(G.facies_codes), max(G.facies_codes)]
#----------------------------------------------------------------------------
# Paths.
data_dir    = '/scratch/users/suihong/SubwaterFan/DatasetsforGAN_freq_3D_28400x128x128x32/'  # Training data path
result_dir  = '/scratch/users/suihong/SubwaterFan/TrainedModels/'  # result data path
#----------------------------------------------------------------------------
# settings for schedual of training
sched.minibatch_dict           = {4: 256, 8: 256, 16: 256, 32: 256, 64: 128, 128: 64} # 64: 16 
sched.G_lrate_dict             = {4: 0.005, 8: 0.005, 16: 0.005, 32: 0.0035, 64: 0.0035, 128: 0.0035} #, 64: 0.0025}; 
sched.D_lrate_dict             = EasyDict(sched.G_lrate_dict); 
sched.lod_training_kimg_dict   = {4: 960, 8:960, 16:960, 32:1280, 64:1280, 128: 1280}
sched.lod_transition_kimg_dict = {4: 960, 8:960, 16:960, 32:1280, 64:1280, 128: 1280}
sched.max_minibatch_per_gpu    = {32: 64, 64: 32, 128: 16}
sched.tick_kimg_dict           = {4: 320, 8:320, 16:320, 32:320, 64:320, 128: 60} 

#----------------------------------------------------------------------------
G_loss.orig_weight = 1   # weight for original Wasserstein GAN loss

#----------------------------------------------------------------------------
# size of input latent vectors and label cubes:
# G.latent_size_z       = 4            # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
# G.latent_size_x       = 4
# G.latent_size_y       = 4
#----------------------------------------------------------------------------
# Settings for condition to global features

cond_label                = False
labeltypes                = [1]  # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity'; but the loss for channel orientation has not been designed in loss.py.
G_loss.MudProp_weight     = 0.2, 
G_loss.Width_weight       = 0.2, 
G_loss.Sinuosity_weight   = 0.2,
if cond_label: desc += '-CondMud_0.2';           # Supplement descriptions onto the folder name if label condition is used    

dataset.cond_label        = cond_label   # Set whether label condition is used
train.cond_label          = cond_label
G.cond_label              = cond_label
G_loss.cond_label         = cond_label
D_loss.cond_label         = cond_label
dataset.labeltypes        = labeltypes
G_loss.labeltypes         = labeltypes

#----------------------------------------------
# Settings for condition to well facies data
cond_well                    = False
G_loss.Wellfaciesloss_weight = 0.7
if cond_well: desc += '-CondWell_0.7';          
dataset.well_enlarge = True;                # set for whether sparse well facies data enlarged, i.e., well point occupies 4x4 cells from 1x1 cell
if cond_well and dataset.well_enlarge: desc += '-Enlarg';  # description of well enlargement onto the folder name of result.

dataset.cond_well            = cond_well
train.cond_well              = cond_well
G.cond_well                  = cond_well
G_loss.cond_well             = cond_well
D_loss.cond_well             = cond_well

#----------------------------------------------
# Settings for condition to probability data
cond_prob                   = False
G_loss.Probcubeloss_weight  = 0.001
G_loss.batch_multiplier     = 4
if cond_prob: desc += '-CondProb_0.0000001'; 
         
dataset.cond_prob           = cond_prob
train.cond_prob             = cond_prob
G.cond_prob                 = cond_prob
G_loss.cond_prob            = cond_prob
D_loss.cond_prob            = cond_prob

#----------------------------------------------
# Setting if loss normalization (into standard Gaussian) is used 
G_loss.lossnorm = False

#----------------------------------------------
# Set if no growing, i.e., the conventional training method. Can be used only if global features are conditioned.
# desc += '-nogrowing-test'; 
# sched.lod_training_kimg_dict   = {4: 0, 8:0, 16:0, 32:0, 64:0, 128: 0}
# sched.lod_transition_kimg_dict = {4: 0, 8:0, 16:0, 32:0, 64:0, 128: 0}
