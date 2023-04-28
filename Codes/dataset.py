import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tfutil

#----------------------------------------------------------------------------
# Parse individual cube from a tfrecords file.

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([4], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

def parse_tfrecord_tf_float16(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([4], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.float16)
    return tf.reshape(data, features['shape']) 

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

def parse_tfrecord_np_float16(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.float16).reshape(shape)

#----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.

class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,               # Directory containing a collection of tfrecords files.
        cond_well       = False,    # Whether condition to well facies data.
        cond_prob       = False,    # Whether condition to probability maps.
        cond_label      = False,    # Whether condition to given global features (labels).

        labeltypes      = [],     # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity'
        well_enlarge    = False,    # If enlarged well points are outputted  
        data_range      = [0, 2],   # Data range of training dataset
        
        repeat          = True,     # Repeat dataset indefinitely.
        shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        num_threads     = 2):       # Number of concurrent threads.
        
        self.tfrecord_dir       = tfrecord_dir
        self.resolution_x       = None
        self.resolution_y       = None
        self.resolution_z       = None
        self.resolution_x_log2  = None
        self.shape              = []        
        self.dtype              = 'uint8'
        self.dynamic_range      = data_range
        self.cond_well          = cond_well
        self.cond_prob          = cond_prob   
        self.cond_label         = cond_label
        self.label_types        = labeltypes
        self.label_size         = len(labeltypes)
        self.label_dtype        = None
        self._np_labels         = None
        self._tf_labels_dataset = None        
        self._tf_labels_var     = None                
        self.well_enlarge       = well_enlarge
        self._tf_wellfacies_dataset = None 
        self._tf_probcubes_dataset = None            
        self._tf_minibatch_in   = None   
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._cur_minibatch     = -1
        self._cur_lod           = -1  
        self.out_name_list      = ['real']  
        self._tf_minibatch_np   = None

        
        if self.cond_label: self.out_name_list.append('label')
        if self.cond_prob: self.out_name_list.append('prob')
        if self.cond_well: self.out_name_list.append('well')
        
        # List realcube tfrecords files and inspect their shapes.
        assert os.path.isdir(self.tfrecord_dir)
        tfr_realcube_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*Data-1r0*.tfrecords'))) # sort the real cube files based on the name pattern
        assert len(tfr_realcube_files) >= 1
        tfr_realcube_shapes = []  # tfr_realcube_shapes
        for tfr_realcube_file in tfr_realcube_files:  # 
            tfr_realcube_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for record in tf.python_io.tf_record_iterator(tfr_realcube_file, tfr_realcube_opt):
                tfr_realcube_shapes.append(parse_tfrecord_np(record).shape)
                break
        # Determine shape and resolution of realcube. some parameters are marked with _realcube_, but some are not. All probcube related parameters are marked with _probcube_.
        self.shape = max(tfr_realcube_shapes, key=lambda shape: np.prod(shape))
        self.resolution_x, self.resolution_y, self.resolution_z = self.shape[1:4]
        self.resolution_x_log2 = int(np.log2(self.resolution_x))
        tfr_realcube_lods_x = [self.resolution_x_log2 - int(np.log2(shape[1])) for shape in tfr_realcube_shapes]     
        
        # Build TF expressions for real cubes and labels (global features).
        with tf.name_scope('Dataset_realcubes_labels'), tf.device('/cpu:0'):
            # Labels (global features)
            if self.cond_label == True:
                # Autodetect label filename.
                tfr_label_file = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.labels')))[0]
                # Load labels.
                self._np_labels = np.zeros([1<<17, 0], dtype=np.float32)
                self._np_labels = np.load(tfr_label_file)[:, labeltypes]                  
                self.label_dtype = self._np_labels.dtype.name
                # Build TF expressions.
                tf_labels_init = tf.zeros(self._np_labels.shape, self._np_labels.dtype)
                self._tf_labels_var = tf.Variable(tf_labels_init, name='labels_var')
                tfutil.set_vars({self._tf_labels_var: self._np_labels})                      
                self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)  
                       
            # Realcubes
            self._tf_minibatch_in = tf.placeholder(tf.int64, name='minibatch_in', shape=[])            
            for tfr_realcube_file, tfr_realcube_shape, tfr_realcube_lod in zip(tfr_realcube_files, tfr_realcube_shapes, tfr_realcube_lods_x):
                if tfr_realcube_lod < 0:
                    continue
                dset = tf.data.TFRecordDataset(tfr_realcube_file, compression_type='', buffer_size=buffer_mb<<17)
                dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
                
                # Realcubes and labels
                dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset)) if self.cond_label == True else tf.data.Dataset.zip((dset,))   
                bytes_per_item = np.prod(tfr_realcube_shape) * np.dtype(self.dtype).itemsize  
                if shuffle_mb > 0:
                    dset = dset.shuffle(((shuffle_mb << 17) - 1) // bytes_per_item + 1)
                if repeat:
                    dset = dset.repeat()
                if prefetch_mb > 0:
                    dset = dset.prefetch(((prefetch_mb << 17) - 1) // bytes_per_item + 1)
                dset = dset.batch(self._tf_minibatch_in)
                self._tf_datasets[tfr_realcube_lod] = dset
            self._tf_iterator = tf.data.Iterator.from_structure(self._tf_datasets[0].output_types, self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) for lod, dset in self._tf_datasets.items()}   
        
        if self.cond_prob or self.cond_well: 
            # Build TF expressions.
            with tf.name_scope('Dataset_prob_well'), tf.device('/cpu:0'): 
                prob_well_dset = ()
                if self.cond_prob == True:
                    # List probcube tfrecord files and inspect its shape.
                    tfr_probcube_file = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*prob*.tfrecords')))[0]
                    tf_probcubes_dset = tf.data.TFRecordDataset(tfr_probcube_file, compression_type='', buffer_size=buffer_mb<<17)
                    tf_probcubes_dset = tf_probcubes_dset.map(parse_tfrecord_tf_float16, num_parallel_calls=num_threads)
                    self._tf_probcubes_dataset = tf_probcubes_dset
                    prob_well_dset = prob_well_dset + (self._tf_probcubes_dataset, )
                if self.cond_well == True:
                    # List well facies tfrecord files and inspect its shape.
                    tfr_wellfacies_file = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*well*.tfrecords')))[0]
                    tf_wellfacies_dset = tf.data.TFRecordDataset(tfr_wellfacies_file, compression_type='', buffer_size=buffer_mb<<17)
                    tf_wellfacies_dset = tf_wellfacies_dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)             
                    self._tf_wellfacies_dataset = tf_wellfacies_dset 
                    prob_well_dset = prob_well_dset + (self._tf_wellfacies_dataset, )
                    
                self._tf_probcubes_wellfacies_dset = tf.data.Dataset.zip(prob_well_dset) 
                if shuffle_mb > 0:
                    self._tf_probcubes_wellfacies_dset = self._tf_probcubes_wellfacies_dset.shuffle(((shuffle_mb << 17) - 1) // bytes_per_item + 1)
                if repeat:
                    self._tf_probcubes_wellfacies_dset = self._tf_probcubes_wellfacies_dset.repeat()
                if prefetch_mb > 0:
                    self._tf_probcubes_wellfacies_dset = self._tf_probcubes_wellfacies_dset.prefetch(((prefetch_mb << 17) - 1) // bytes_per_item + 1)         
                self._tf_probcubes_wellfacies_dset = self._tf_probcubes_wellfacies_dset.batch(self._tf_minibatch_in)
                self._tf_probcubes_wellfacies_iterator = tf.data.Iterator.from_structure(self._tf_probcubes_wellfacies_dset.output_types,  self._tf_probcubes_wellfacies_dset.output_shapes)
                self._tf_probcubes_wellfacies_init_ops = self._tf_probcubes_wellfacies_iterator.make_initializer(self._tf_probcubes_wellfacies_dset)              

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            if self.cond_well or self.cond_prob: self._tf_probcubes_wellfacies_init_ops.run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => cubes, labels, probcubes, wellfacies
        out = []     
        real_or_label_tup = self._tf_iterator.get_next()  # tuple of: (real, labels)
        for i in range(len(real_or_label_tup)): out.append(real_or_label_tup[i])
        if self.cond_well or self.cond_prob: 
            prob_or_well_tup = self._tf_probcubes_wellfacies_iterator.get_next()  # tuple of 6 dim: (prob, well)
            for i in range(len(prob_or_well_tup)): out.append(prob_or_well_tup[i])
        if self.cond_well and self.well_enlarge: 
            # 1) to enlarge area influenced by well facies; 2) since cpu device only accept max_pool opt with data_format = 'NHWC' instead of 'NCHW', so transpose twice to deal with that
            wellfacies = out[-1]
            wellfacies = tf.cast(wellfacies, tf.float16)
            wellfacies = tf.transpose(wellfacies, perm=[0, 2, 3, 4, 1])
            wellfacies = tf.nn.max_pool3d(wellfacies, ksize = [1,4,4,4,1], strides=[1,1,1,1,1], padding='SAME', data_format='NDHWC') 
            wellfacies = tf.transpose(wellfacies, perm=[0, 4, 1, 2, 3])  
            out[-1] = wellfacies                
        out_dict = {self.out_name_list[i]: out[i] for i in range(len(self.out_name_list))} 
        return out_dict
    
    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => cubes, labels, probcubes
        self.configure(minibatch_size, lod)
        if self._tf_minibatch_np is None:
            self._tf_minibatch_np = self.get_minibatch_tf()
        return tfutil.run(self._tf_minibatch_np)    
    
    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_cubeorlabel_tf(self): # => cubes, labels
        out = []     
        real_or_label_tup = self._tf_iterator.get_next()  # tuple of: (real, labels)
        for i in range(len(real_or_label_tup)): out.append(real_or_label_tup[i])
        out_dict = {}
        out_dict['real'] = out[0]
        if self.cond_label: out_dict['label'] = out[-1]
        return out_dict

    # Get next minibatch as NumPy arrays.
    def get_minibatch_cubeorlabel_np(self, minibatch_size, lod=0): # => cubes, labels
        self.configure(minibatch_size, lod)
        return tfutil.run(self.get_minibatch_cubeorlabel_tf())
        
     # Get next minibatch as TensorFlow expressions.
    def get_minibatch_proborwell_tf(self): # => probcubes, wellfacies
        assert self.cond_prob or self.cond_well
        out = []    
        prob_or_well_tup = self._tf_probcubes_wellfacies_iterator.get_next()  # tuple of 6 dim: (prob, well)
        for i in range(len(prob_or_well_tup)): out.append(prob_or_well_tup[i])  
        if self.cond_well and self.well_enlarge:           
            # 1) to enlarge area influenced by well facies; 2) since cpu device only accept max_pool opt with data_format = 'NHWC' instead of 'NCHW', so transpose twice to deal with that
            wellfacies = out[-1]
            wellfacies = tf.cast(wellfacies, tf.float16)
            wellfacies = tf.transpose(wellfacies, perm=[0, 2, 3, 4, 1])
            wellfacies = tf.nn.max_pool3d(wellfacies, ksize = [1,4,4,4,1], strides=[1,1,1,1,1], padding='SAME', data_format='NDHWC') 
            wellfacies = tf.transpose(wellfacies, perm=[0, 4, 1, 2, 3]) 
            out[-1] = wellfacies
        out_dict = {}
        if self.cond_prob: out_dict['prob'] = out[0]
        if self.cond_well: out_dict['well'] = out[-1]
        return out_dict

    # Get next minibatch as NumPy arrays.
    def get_minibatch_proborwell_np(self, minibatch_size, lod=0): # => probcubes, wellfacies
        self.configure(minibatch_size, lod)
        return tfutil.run(self.get_minibatch_proborwell_tf())      

    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size): # => labels
        return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 0, self._np_labels.shape[0], dtype=tf.int32))       

#----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.

def load_dataset(class_name='dataset.TFRecordDataset', data_dir=None, verbose=False, **kwargs):
    adjusted_kwargs = dict(kwargs)
    if 'tfrecord_dir' in adjusted_kwargs and data_dir is not None:
        adjusted_kwargs['tfrecord_dir'] = os.path.join(data_dir, adjusted_kwargs['tfrecord_dir'])
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = tfutil.import_obj(class_name)(**adjusted_kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
        print('Label size    =', dataset.label_size)
    return dataset
