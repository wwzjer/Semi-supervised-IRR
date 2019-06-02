import os
import h5py
import numpy as np
import tensorflow as tf
import cv2
import random
from utils import total_variation
import skimage.measure

##################### select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################
#os.environ['CUDA_VISIBLE_DEVICES'] = str(monitoring_gpu.GPU_INDEX)
############################################################################


####Delete all flags before declare#####

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)
tf.reset_default_graph()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_h5_file', 50000,"""number of training h5 files.""")
tf.app.flags.DEFINE_integer('num_patches', 20,"""number of patches in each h5 file.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,"""learning rate.""")
tf.app.flags.DEFINE_integer('epoch', 15,"""epoch.""")
tf.app.flags.DEFINE_integer('batch_size', 20,"""Batch size.""")
tf.app.flags.DEFINE_integer('num_channels', 3,"""Number of channels of the input.""")
tf.app.flags.DEFINE_integer('image_size', 64,"""Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 64,"""Size of the labels.""")
tf.app.flags.DEFINE_string("data_path", "./data/h5data/", "The path of synthesized data")
tf.app.flags.DEFINE_string("data_path_real","./data/h5data_real/","The path of real data")
tf.app.flags.DEFINE_string("save_model_path", "./model/", "The path of saving model")
tf.app.flags.DEFINE_float('lambda1',1.,"lambda1 value")
tf.app.flags.DEFINE_float('lambda2',.5,"lambda2 value")
tf.app.flags.DEFINE_float('lambda3',1e-5,"lambda3 value")
tf.app.flags.DEFINE_float('lambda4',1e-9,"lambda4 value")
tf.app.flags.DEFINE_integer('num_components', 3, """"number of gaussian mixture components""")



# read h5 files
def read_data(file):
  with h5py.File(file, 'r') as hf:
    data = hf.get('data')
    label = hf.get('label')
    return np.array(data), np.array(label)



# guided filter
def guided_filter(data, num_patches = FLAGS.num_patches, width = FLAGS.image_size, height = FLAGS.image_size, channel = FLAGS.num_channels):
    r = 15
    eps = 1.0
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            I = data[i, :, :,j]
            p = data[i, :, :,j]
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps) 
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            q = mean_a * I + mean_b 
            batch_q[i, :, :,j] = q 
    return batch_q



# initialize weights
def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables


# network structure
def inference(images,detail,factor=1):

   #  layer 1
   with tf.variable_scope('conv_1'):
      kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels, 16])
      biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases_1')      
      scale = tf.Variable(tf.ones([16]), trainable=True, name='scale_1')
      beta = tf.Variable(tf.zeros([16]), trainable=True, name='beta_1')
  
      conv = tf.nn.conv2d(detail, kernel, [1, 1, 1, 1], padding='SAME')
      #stride [1,x_movement,y_movement,1]
      #padding 'VALID' small; 'SAME' equal
      feature = tf.nn.bias_add(conv, biases)
  
      mean, var = tf.nn.moments(feature,[0, 1, 2])
      feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
  
      conv_shortcut = tf.nn.relu(feature_normal)
  
   #  layers 2 to 25
   for i in range(12):
     with tf.variable_scope('conv_%s'%(i*2+2)):
       kernel = create_kernel(name=('weights_%s'%(i*2+2)), shape=[3, 3, 16, 16])
       biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name=('biases_%s'%(i*2+2)))
       scale = tf.Variable(tf.ones([16]), trainable=True, name=('scale_%s'%(i*2+2)))
       beta = tf.Variable(tf.zeros([16]), trainable=True, name=('beta_%s'%(i*2+2)))   
   
       conv = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')     
       feature = tf.nn.bias_add(conv, biases)
   
       mean, var = tf.nn.moments(feature,[0, 1, 2])
       feature_normal = tf.nn.batch_normalization(feature,  mean, var, beta, scale, 1e-5)
   
       feature_relu = tf.nn.relu(feature_normal)

  
     with tf.variable_scope('conv_%s'%(i*2+3)): 
       kernel = create_kernel(name=('weights_%s'%(i*2+3)), shape=[3, 3, 16, 16])
       biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name=('biases_%s'%(i*2+3)))
       scale = tf.Variable(tf.ones([16]), trainable=True, name=('scale_%s'%(i*2+3)))
       beta = tf.Variable(tf.zeros([16]), trainable=True, name=('beta_%s'%(i*2+3)))   
   
       conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')     
       feature = tf.nn.bias_add(conv, biases)
  
       mean, var  = tf.nn.moments(feature,[0, 1, 2])
       feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
   
       feature_relu = tf.nn.relu(feature_normal)  

       conv_shortcut = tf.add(conv_shortcut, feature_relu)  #  shortcut

  
   # layer 26
   with tf.variable_scope('conv_26'):
      kernel = create_kernel(name='weights_26', shape=[3, 3, 16, FLAGS.num_channels])   
      biases = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True, name='biases_26')
      scale = tf.Variable(tf.ones([3]), trainable=True, name=('scale_26'))
      beta = tf.Variable(tf.zeros([3]), trainable=True, name=('beta_26'))
  
      conv = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
      feature = tf.nn.bias_add(conv, biases)
 
      mean, var  = tf.nn.moments(feature,[0, 1, 2])
      neg_residual = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
 
      final_out = tf.add(images, neg_residual/factor)

   return final_out

# GMM initialization
def initialize(X,k):
    n = X.shape[1]
    idx = random.sample(range(n),k)
    m = X[:,idx]
    a = m.T*X-np.square(m).T/2
    #R = np.argmax(a,0)
    R = list(map(lambda x: x==np.max(x,0),a.T)) * np.ones(shape=a.T.shape)
    return R

# E step
def expectation(X,model):
    mu = model['mu']
    Sigma = model['Sigma']
    w = model['weight']

    n = X.shape[1]
    k = mu.shape[1]
    logRho = np.zeros([n,k])

    for i in range(k):
        logRho[:,i] = loggausspdf(X,mu[0,i],Sigma[0,i])

    logRho = logRho+np.log(w)
    T = logsumexp(logRho)
    llh = np.sum(T)/n
    logR = logRho.T-T
    R = np.exp(logR).T
    return (R,llh)

def loggausspdf(X,mu,Sigma):
    d = X.shape[0]
    X = X-mu
    U = np.sqrt(Sigma)
    Q = X/U
    q = np.square(Q)
    c = d*np.log(2*np.pi)+2*np.sum(np.log(U))
    y = -(c+q)/2
    return y

def logsumexp(x):
    y = np.max(x,1)
    x = x.T-y
    s = y+np.log(np.sum(np.exp(x),0))
    return s

# M step
def maximizationModel(X,R):
    k = R.shape[1]
    nk = np.sum(R,0)
    mu = np.zeros([1,k])

    w = nk/R.shape[0]
    Sigma = np.zeros([1,k])
    sqrtR = np.sqrt(R)
    for i in range(k):
        Xo = X-mu[0,i]
        Xo = Xo*sqrtR[:,i]
        Sigma[:,i] = np.dot(Xo,Xo.T)/nk[i]
        Sigma[:,i] = Sigma[:,i]+1e-6

    model = {'mu':mu, 'Sigma':Sigma, 'weight':w}
    return model



if __name__ == '__main__':
    
    images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))  # supervised data (None,64,64,3)
    details = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels)) # supervised detail layer (None,64,64,3)
    labels = tf.placeholder(tf.float32, shape=(None, FLAGS.label_size, FLAGS.label_size, FLAGS.num_channels))  # supervised label (None,64,64,3)
    outputs = inference(images,details) # supervised outputs (None,64,64,3)
    loss1 = tf.reduce_mean(tf.square(labels - outputs))    # supervised MSE loss

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        tf.get_variable_scope().reuse_variables()
        un_images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))  # unsupervised data (None,64,64,3)
        un_details = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels)) # unsupervised detail layer (None,64,64,3)
        un_outputs = inference(un_images,un_details)  # unsupervised outputs (None,64,64,3)

    R = tf.placeholder(tf.float32,shape=(None,FLAGS.num_components))

    #mu = tf.placeholder(tf.float32,shape=(1,3))
    Sigma = tf.placeholder(tf.float32,shape=(1,FLAGS.num_components),name='Sigma')
    weight = tf.placeholder(tf.float32,shape=(FLAGS.num_components),name='weight')
    
    loss2 = 0.
    for k in range(FLAGS.num_components):
        loss2 += R[:,k]*tf.reshape(tf.square(un_images-un_outputs),[1,-1])*.5/Sigma[0,k]
    loss2 = tf.reduce_mean(loss2)
    
    loss3 = tf.reduce_mean(total_variation(un_outputs))
    
    tfd = tf.contrib.distributions
    mean, var = tf.nn.moments(tf.reshape(images - labels,(-1,1)),[0])
    f1 = tfd.Normal(loc=mean, scale=tf.sqrt(var))
    
    loss4 = []
    for k in range(FLAGS.num_components):
        loss4.append(tfd.kl_divergence(f1, tfd.Normal(loc=0., scale=Sigma[0,k])))

    loss4 = tf.reduce_min(loss4)
    loss = FLAGS.lambda1*loss1 + FLAGS.lambda2*loss2 + FLAGS.lambda3*loss3 + FLAGS.lambda4*loss4


    lr_ = FLAGS.learning_rate
    lr  = tf.placeholder(tf.float32 ,shape = [])
    g_optim =  tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver(max_to_keep = 5)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # GPU setting
    config.gpu_options.allow_growth = True


    epoch = int(FLAGS.epoch)

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        validation_data_name = "validation.h5"
        validation_h5_data, validation_h5_label = read_data(FLAGS.data_path + validation_data_name)

        validation_data = validation_h5_data
        validation_data = np.transpose(validation_data, (0,2,3,1))   # image data

        validation_detail = validation_data - guided_filter(validation_data)  # detail layer

        validation_label = np.transpose(validation_h5_label, (0,2,3,1)) # label

        print("Start training")
        start_point = 0


        for j in range(start_point,epoch):  

            if j+1 >(epoch/3):  # reduce learning rate
                lr_ = FLAGS.learning_rate*0.1
            if j+1 >(2*epoch/3):
                lr_ = FLAGS.learning_rate*0.01

            for num in range(FLAGS.num_h5_file):    # h5 files                
                train_data_name = "train" + str(num+1) + ".h5"
                train_h5_data, train_h5_label = read_data(FLAGS.data_path + train_data_name)

                train_data = np.transpose(train_h5_data, (0,2,3,1))   # image data
                detail_data = train_data - guided_filter(train_data)  # detail layer
                train_label = np.transpose(train_h5_label, (0,2,3,1)) # label

                train_real_data_name = "real" + str(num+1) + ".h5"
                train_real_h5_data, _ = read_data(FLAGS.data_path_real + train_real_data_name)

                train_real_data = np.transpose(train_real_h5_data,(0,2,3,1))
                detail_real_data = train_real_data - guided_filter(train_real_data)

                if j==0:
                    R_real = initialize(train_real_data.reshape(1,-1),FLAGS.num_components)
                    model = {'mu':np.zeros([1,FLAGS.num_components],dtype=np.float32),
                             'Sigma':np.random.rand(1,FLAGS.num_components).astype(np.float32),
                             'weight':np.sum(R_real,0)/float(np.size(R_real,0))}
                    Error = train_real_data.reshape(1, -1)
                else:
                    unsupervised_outputs = sess.run(un_outputs, feed_dict={un_images:train_real_data, un_details: detail_real_data})
                    Error = train_real_data.reshape(1,-1) - unsupervised_outputs.reshape(1,-1)

                R_real,_ = expectation(Error,model)
                model = maximizationModel(Error,R_real)
                R_real, _ = expectation(Error,model)


                _,lossvalue = sess.run([g_optim,loss],
                                       feed_dict={images:train_data,
                                                  details:detail_data,
                                                  labels:train_label,
                                                  un_images:train_real_data,
                                                  un_details:detail_real_data,
                                                  lr:lr_,R:R_real,
                                                  Sigma:model['Sigma'],
                                                  weight:model['weight']})
                pred_im = sess.run(outputs, feed_dict={images:train_data, details: detail_data})
                psnr = skimage.measure.compare_psnr(train_label,pred_im)
                


            model_name = 'model-epoch'   # save model
            save_path_full = os.path.join(FLAGS.save_model_path, model_name)
            saver.save(sess, save_path_full, global_step = j+1)



            Validation_Loss  = sess.run(loss,  
                                        feed_dict={images: validation_data, 
                                                   details: validation_detail, 
                                                   labels:validation_label,
                                                   un_images:train_real_data,
                                                   un_details:detail_real_data,
                                                   lr:lr_,R:R_real,
                                                   Sigma:model['Sigma'],
                                                   weight:model['weight']})
            pred_val = sess.run(outputs, feed_dict={images:validation_data, details: validation_detail})
            psnr_val = skimage.measure.compare_psnr(validation_label,pred_val)

            print ('Epoch %d :Train loss %.4f, Validation loss %.4f, Train PSNR %.4f, Validation PSNR %.4f' %
                   (j+1, lossvalue, Validation_Loss, psnr, psnr_val))
