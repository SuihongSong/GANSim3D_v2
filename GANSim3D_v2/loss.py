import numpy as np
import tensorflow.compat.v1 as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def gaussian_kernel(size: int, mean: float, std: float,):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)

#----------------------------------------------------------------------------
# Generator loss function.

#** Only the labels inputted into G is of the form of cube (same size as latent vectors); labels from D is still of form [None, label size]


def G_wgan_acgan(G, D, lod, 
                 labels, well_facies, prob_cubes, 
                 minibatch_size, 
                 cond_well           = False,    # Whether condition to well facies data.
                 cond_prob           = False,    # Whether condition to probability maps.
                 cond_label          = False,    # Whether condition to given global features (labels).
                 Wellfaciesloss_weight = 0.7, 
                 MudProp_weight = 0.2, 
                 Width_weight = 0.2, 
                 Sinuosity_weight = 0.2, 
                 orig_weight = 2, 
                 labeltypes = None, 
                 Probcubeloss_weight = 0.0000001, 
                 batch_multiplier = 4, 
                 lossnorm = True): 
    #labeltypes, e.g., labeltypes = [1]  # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity', set in config file
    # loss for channel orientation is not designed below, so do not include "0" in labeltypes.
    # lossnorm: True to normalize loss into standard Gaussian before multiplying with weights.   
    
    if cond_prob:
        prob_cubes = tf.cast(prob_cubes, tf.float32)
        prob_cubes_lg = tf.reshape(tf.tile(tf.expand_dims(prob_cubes, 1), [1, batch_multiplier, 1, 1, 1, 1]), ([-1] + G.input_shapes[3][1:]))   
    else:
        prob_cubes_lg = tf.zeros([0] + G.input_shapes[3][1:])
        batch_multiplier = 1
    
    if cond_label:
        label_size = len(labeltypes)
        labels_list = []
        for k in range(label_size):
            labels_list.append(tf.random.uniform(([minibatch_size]), minval=-1, maxval=1))
        if 1 in labeltypes:   # mud proportion
            ind = labeltypes.index(1)
            labels_list[ind] = tf.clip_by_value(labels[:, ind] + tf.random.uniform([minibatch_size], minval=-0.2, maxval=0.2), -1, 1)    
        labels_in = tf.stack(labels_list, axis = 1)   
        labels_lg = tf.reshape(tf.tile(tf.expand_dims(labels_in, 1), [1, batch_multiplier, 1]), ([-1] + [G.input_shapes[1][1].value]))
        labels_lg_cube = tf.expand_dims(tf.expand_dims(tf.expand_dims(labels_lg, -1), -1), -1)
        labels_lg_cube = tf.tile(labels_lg_cube, [1,1,G.input_shapes[1][-3], G.input_shapes[1][-2], G.input_shapes[1][-1]])
    else: 
        labels_lg_cube = tf.zeros([0] + G.input_shapes[1][1:])
        
    if cond_well:
        well_facies = tf.cast(well_facies, tf.float32)
        well_facies_lg = tf.reshape(tf.tile(tf.expand_dims(well_facies, 1), [1, batch_multiplier, 1, 1, 1, 1]), ([-1] + G.input_shapes[2][1:]))
    else:
        well_facies_lg = tf.zeros([0] + G.input_shapes[2][1:])
        
    latents = tf.random_normal([minibatch_size * batch_multiplier] + G.input_shapes[0][1:])
 
    fake_cubes_out = G.get_output_for(latents, labels_lg_cube, well_facies_lg, prob_cubes_lg, is_training=True)  
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_cubes_out, is_training=True))
    loss = -fake_scores_out
    if lossnorm: loss = (loss -211.2312) / 55.90123   #To Normalize
    loss = tfutil.autosummary('Loss_G/GANloss', loss)
    loss = loss * orig_weight     

    if cond_label:       
        with tf.name_scope('LabelPenalty'):
            def addMudPropPenalty(index):
                MudPropPenalty = tf.nn.l2_loss(labels_lg[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
                if lossnorm: MudPropPenalty = (MudPropPenalty -0.36079434843794) / 0.11613414177144  # To normalize this loss 
                MudPropPenalty = tfutil.autosummary('Loss_G/MudPropPenalty', MudPropPenalty)        
                MudPropPenalty = MudPropPenalty * MudProp_weight  
                return loss+MudPropPenalty
            if 1 in labeltypes:
                ind = labeltypes.index(1)
                loss = addMudPropPenalty(ind)
            
            def addWidthPenalty(index):
                WidthPenalty = tf.nn.l2_loss(labels_lg[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
                if lossnorm: WidthPenalty = (WidthPenalty -0.600282781464712) / 0.270670509379704  # To normalize this loss 
                WidthPenalty = tfutil.autosummary('Loss_G/WidthPenalty', WidthPenalty)             
                WidthPenalty = WidthPenalty * Width_weight            
                return loss+WidthPenalty
            if 2 in labeltypes:
                ind = labeltypes.index(2)
                loss = tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addWidthPenalty(ind), lambda: loss)
            
            def addSinuosityPenalty(index):
                SinuosityPenalty = tf.nn.l2_loss(labels_lg[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
                if lossnorm: SinuosityPenalty = (SinuosityPenalty -0.451279248935835) / 0.145642580091667  # To normalize this loss 
                SinuosityPenalty = tfutil.autosummary('Loss_G/SinuosityPenalty', SinuosityPenalty)            
                SinuosityPenalty = SinuosityPenalty * Sinuosity_weight              
                return loss+SinuosityPenalty
            if 3 in labeltypes:
                ind = labeltypes.index(3)
                loss = tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addSinuosityPenalty(ind), lambda: loss)  
    if cond_well:            
        def Wellpoints_L2loss(well_facies, fake_cubes):
            loss = tf.nn.l2_loss(well_facies[:,0:1]* (well_facies[:,1:2] - (fake_cubes+1)/2))
            loss = loss / tf.reduce_sum(well_facies[:, 0:1])
            return loss
        def addwellfaciespenalty(well_facies, fake_cubes_out, loss, Wellfaciesloss_weight):
            with tf.name_scope('WellfaciesPenalty'):
                WellfaciesPenalty =  Wellpoints_L2loss(well_facies, fake_cubes_out)       
                if lossnorm: WellfaciesPenalty = (WellfaciesPenalty - 0.001028) / 0.002742
                WellfaciesPenalty = tfutil.autosummary('Loss_G/WellfaciesPenalty', WellfaciesPenalty)
                loss += WellfaciesPenalty * Wellfaciesloss_weight   
            return loss   
        loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 5.)), lambda: addwellfaciespenalty(well_facies_lg, fake_cubes_out, loss, Wellfaciesloss_weight), lambda: loss)
  
    if cond_prob:
        ## Settings for condition-based loss for probability map are explained in detail in 'Note for settings of probability map-based loss.ipynb'.   
        def addfaciescodeexpectationloss(probs, fakes, weight, batchsize, relzs, loss):  # used when resolution is less than 64x64
            with tf.name_scope('ProbcubePenalty'):
                expects_fake = tf.reduce_mean(tf.reshape(fakes, ([batchsize, relzs] + G.input_shapes[3][1:])), 1)  # code expectation for fakes
                ProbPenalty = tf.nn.l2_loss((probs * 1 + (1-probs)*(-1)) - expects_fake) # (0+1)/2 for channel complex, -1 for mud facies
                if lossnorm: ProbPenalty = ((ProbPenalty*tf.cast(relzs, tf.float32)) - 301346) / 113601   # normalize
                ProbPenalty = tfutil.autosummary('Loss_G/ProbPenalty', ProbPenalty)
            loss += ProbPenalty * weight
            return loss

        def addfaciescodedistributionloss(probs, fakes, weight, batchsize, relzs, loss):  # used when resolution is 64x64        
            with tf.name_scope('ProbcubePenalty'):   
                # In paper, only probability map for channel complex is condisered. If multiple probability maps for multiple facies are considered, needs to calculate channelindicator and probPenalty for each facies.          
                channelindicator = 1 / (1+tf.math.exp(-8*(fakes))) # use adjusted sigmoid to replace thresholding.   
                #channelindicator = tf.where(tf.math.greater(fakes, 0.), tf.fill(tf.shape(fakes), 1.), tf.fill(tf.shape(fakes), 0.))         
                probs_fake = tf.reduce_mean(tf.reshape(channelindicator, ([batchsize, relzs] + G.input_shapes[3][1:])), 1)            
                #****** With Gaussian smoothing
                #kernel = gaussian_kernel(3, 0., 7.)[:, :, tf.newaxis, tf.newaxis] # expend dimensions
                #probs_fake = tf.nn.conv2d(probs_fake, kernel, strides=[1,1,1,1], padding='SAME', data_format='NCHW')  
                #****** Use cross entropy loss
                #probs = (probs * 0.999) + 0.001  # to make sure the probs are larger than 0 and smaller than 1.
                #probs_fake = (probs_fake * 0.999) + 0.001
                #ProbPenalty = tf.math.reduce_mean( - (probs * tf.math.log(probs_fake) + (1 - probs) * tf.math.log(1 - probs_fake)))   #Cross entropy loss
                #****** L2 loss
                ProbPenalty = tf.nn.l2_loss(probs - probs_fake)  # L2 loss
                if lossnorm: ProbPenalty = ((ProbPenalty*tf.cast(relzs, tf.float32))- 301346) / 113601   # normalize
                ProbPenalty = tfutil.autosummary('Loss_G/ProbPenalty', ProbPenalty)
            loss += ProbPenalty * weight
            return loss
        loss = tf.cond(tf.math.less_equal(lod, tf.fill([], -1.)), \
                       lambda: addfaciescodedistributionloss(prob_cubes, fake_cubes_out, Probcubeloss_weight, minibatch_size, batch_multiplier, loss), \
                       lambda: addfaciescodeexpectationloss(prob_cubes, fake_cubes_out, Probcubeloss_weight, minibatch_size, batch_multiplier, loss))        
     
    loss = tfutil.autosummary('Loss_G/Total_loss', loss)    
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_wgangp_acgan(G, D, opt, minibatch_size, reals, labels, well_facies, prob_cubes,
    cond_well       = False,    # Whether condition to well facies data.
    cond_prob       = False,    # Whether condition to probability maps.
    cond_label      = False,    # Whether condition to given global features (labels).                   
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    label_weight    = 10):       # Weight of the conditioning terms.      

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    
    if cond_label:
        labels_cube = tf.expand_dims(tf.expand_dims(tf.expand_dims(labels, -1), -1), -1)
        labels_cube = tf.tile(labels_cube, [1,1,G.input_shapes[0][-3], G.input_shapes[0][-2], G.input_shapes[0][-1]])   
    else:
        labels_cube = tf.zeros([0] + G.input_shapes[1][1:])
    fake_cubes_out = G.get_output_for(latents, labels_cube, well_facies, prob_cubes, is_training=True)    
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_cubes_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss_D/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss_D/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1, 1], 0.0, 1.0, dtype=fake_cubes_out.dtype)
        mixed_cubes_out = tfutil.lerp(tf.cast(reals, fake_cubes_out.dtype), fake_cubes_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_cubes_out, is_training=True))
        #mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_cubes_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3, 4]))
        mixed_norms = tf.reshape(mixed_norms,(-1,1))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    loss = tfutil.autosummary('Loss_D/WGAN_GP_loss', loss)
   
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss_D/epsilon_penalty', tf.square(real_scores_out))
        loss += epsilon_penalty * wgan_epsilon

    if cond_label:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.l2_loss(labels - real_labels_out)                            
            label_penalty_fakes = tf.nn.l2_loss(labels - fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss_D/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss_D/label_penalty_fakes', label_penalty_fakes)
            loss += (label_penalty_reals + label_penalty_fakes) * label_weight
  
    loss = tfutil.autosummary('Loss_D/Total_loss', loss)
    return loss