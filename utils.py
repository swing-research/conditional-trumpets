import tensorflow as tf
import numpy as np
import cv2
import argparse
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

  

def conditional_sampling(pz , inj_model , bij_model , x_test , y_test ,
                         n_average , n_test = 5 , n_sample_show = 4):
    '''Generate posterior samples, MMSE, MAP and UQ'''
    
    def normalization(image , min_x , max_x):
        image += -image.min()
        image /= image.max()
        image *= max_x - min_x
        image += min_x
        
        return image
    
    y_s_single = y_test[1*n_test:2 * n_test]
    y_reshaped = y_test[1*n_test:2 * n_test].numpy()
    if np.shape(y_s_single)[1] != np.shape(x_test)[1]:
        
        r = np.shape(x_test)[1]
        if np.shape(y_test)[3] == 2:
            y_reshaped = np.sqrt(np.sum(np.square(y_reshaped) , axis = 3 , keepdims=True))
            y_reshaped = data_normalization(y_reshaped)
        
        y_reshaped_orig = np.zeros([n_test , np.shape(x_test)[1] , np.shape(x_test)[2] , np.shape(x_test)[3]])
        for i in range(n_test):
            if np.shape(x_test)[3] == 1:
                y_reshaped_orig[i,:,:,0] = cv2.resize(y_reshaped[i][:,:,0] , (r,r),
                                                  interpolation = cv2.INTER_NEAREST)
            else:
                y_reshaped_orig[i] = cv2.resize(y_reshaped[i] , (r,r),
                                            interpolation = cv2.INTER_NEAREST)
         
        y_reshaped = y_reshaped_orig
    
    y_s = tf.repeat(y_s_single, n_average, axis = 0)
    
    gt = x_test[1*n_test:2 * n_test].numpy()
    
    z_random_base = pz.prior.sample(n_average * n_test)
    z_random_base_mean = (z_random_base[:n_test] - pz.mu) * 0 + pz.mu
    
    z_random = bij_model(z_random_base ,
                            y_s,
                            reverse = True)[0] # Intermediate samples
    
    z_random_mean = bij_model(z_random_base_mean ,
                            y_s_single,
                            reverse = True)[0] # Intermediate samples
    
    
    x_sampled = inj_model(z_random,
                      y_s,
                      reverse = True)[0].numpy() 
    
    x_MAP = inj_model(z_random_mean,
                      y_s_single,
                      reverse = True)[0].numpy() 
        
    
    n_sample = n_sample_show + 5 
    final_shape = [n_test*(n_sample), np.shape(x_sampled)[1] , np.shape(x_sampled)[2],
                   np.shape(x_sampled)[3]]
    x_sampled_all = np.zeros(final_shape)
    mean_vec = np.zeros([n_test , np.shape(x_sampled)[1] , np.shape(x_sampled)[2],
                         np.shape(x_sampled)[3]] , dtype = np.float32)
    
    SSIM_MMSE = 0
    SSIM_pseudo = 0
    SSIM_MAP = 0
    for i in range(n_test):
        x_sampled_all[i*n_sample] = gt[i]
        x_sampled_all[i*n_sample + 1] = x_MAP[i]
        x_sampled_all[i*n_sample + 2] = np.mean(x_sampled[i*n_average:i*n_average + n_average] , axis = 0)
        x_sampled_all[i*n_sample+3:i*n_sample + 3 + n_sample_show] = x_sampled[i*n_average:i*n_average + n_sample_show]
        x_sampled_all[i*n_sample + 4 + n_sample_show] = y_reshaped[i]
        x_sampled_all[i*n_sample + 3 + n_sample_show] = normalization(np.std(x_sampled[i*n_average:i*n_average + n_average] , axis = 0) , gt.min() , gt.max())

        mean_vec[i] = np.mean(x_sampled[i*n_average:i*n_average + n_average] , axis = 0)
        SSIM_MMSE = SSIM_MMSE + ssim(mean_vec[i] + 1.0,
                           gt[i] + 1.0,
                           data_range=gt.max() - gt.min(),
                           multichannel=True)
        
        SSIM_pseudo = SSIM_pseudo + ssim(y_reshaped[i] + 1.0,
                           gt[i] + 1.0,
                           data_range=gt.max() - gt.min(),
                           multichannel=True)
        
        SSIM_MAP = SSIM_MAP + ssim(x_MAP[i] + 1.0,
                           gt[i] + 1.0,
                           data_range=gt.max() - gt.min(),
                           multichannel=True)
    
    snr_pseudo = SNR_computation(gt +1.0 , y_reshaped + 1.0 )
    snr_MMSE = SNR_computation(gt + 1.0 , mean_vec + 1.0)
    snr_MAP = SNR_computation(gt + 1.0 , x_MAP + 1.0)

    print('SNR of pseudo inverse:{:.3f}'.format(snr_pseudo))
    print('SNR of MMSE:{:.3f}'.format(snr_MMSE))
    print('SNR of MAP:{:.3f}'.format(snr_MAP))
    print('SSIM of pseudo inverse:{:.3f}'.format(SSIM_pseudo/n_test))
    print('SSIM of MMSE:{:.3f}'.format(SSIM_MMSE/n_test))
    print('SSIM of MAP:{:.3f}'.format(SSIM_MAP/n_test))
    return x_sampled_all , y_s_single.numpy() , snr_MMSE,  SSIM_MMSE/n_test




def classified_sampling(pz , inj_model , bij_model , num_classes):
    '''class-based image generation'''
    
    num_samples = 5
    labels = np.zeros([num_classes * num_samples , num_classes])
    for i in range(num_classes):
        
        labels[i*num_samples:(i+1)*num_samples , i] = 1
    
    labels = tf.convert_to_tensor(labels , dtype = tf.float32)
    
    z_random_base = pz.prior.sample(num_samples * num_classes) # sampling from base (gaussian) with Temprature = 1
    z_random = bij_model(z_random_base ,
                            labels,
                            reverse = True)[0] # Intermediate samples with Temprature = 1
    
    x_sampled = inj_model(z_random,
                      labels,
                      reverse = True)[0].numpy() # Samples with Temprature = 1

    return x_sampled



def data_normalization(x):
    '''Normalize data between -1 and 1'''
    x = x.astype('float32')
    x = x - (x.max() + x.min())/2
    x /= (x.max())
    
    return x


def image_resizer(x , r):
    '''Resize images ina given resolution'''
    b , h, _ , nch = np.shape(x)
    y = np.zeros((np.shape(x)[0], r, r, nch))
    
    if x.shape[1] != r:

        for i in range(b):
            if nch == 1:
                y[i,:,:,0] = cv2.resize(x[i] , (r,r))
            else:
                y[i] = cv2.resize(x[i] , (r,r))           
    else:
        y = x
        
    return y



def SNR_computation(x_true , x_pred):
    '''Calculate SNR of a barch of true and their estimations'''
        
    x_true = np.reshape(x_true , [np.shape(x_true)[0] , -1])
    x_pred = np.reshape(x_pred , [np.shape(x_pred)[0] , -1])
    
    Noise = x_true - x_pred
    Noise_power = np.sum(np.square(np.abs(Noise)), axis = -1)
    Signal_power = np.sum(np.square(np.abs(x_true)) , axis = -1)
    SNR = 10*np.log10(np.mean(Signal_power/Noise_power))
  
    return SNR
    


def traveltime_op(x_train , x_test , noise_snr = 100):
    '''Linearized traveltime tomography operator'''
    
    folder = 'datasets/traveltime/'

    A = np.load(folder + '10' + '_forward.npy')
    A_pinv = np.load(folder + '10' + '_pinverse.npy')
    
    white = np.ones([1024 , 1])
    white_hat = A_pinv @ (A @ white)
    white_hat = 1-white_hat
    cv2.imwrite(folder + 'sensors_locations.png' , white_hat.reshape([32,32,1]) * 255)
    
    b_train , image_size , _ , c = tf.shape(x_train)
    b_test = tf.shape(x_test)[0]

    y_train = np.matmul(np.reshape(x_train, (b_train, -1)), 
            A.T)
    y_test = np.matmul(np.reshape(x_test, (b_test, -1)), 
            A.T)
    
    noise_sigma_train = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
        np.square(np.reshape(y_train, (b_train , -1))) , -1)))

    
    noise_sigma_test = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
        np.square(np.reshape(y_test, (b_test , -1))) , -1)))
    
    
    n_train = np.random.normal(loc = 0 , scale = noise_sigma_train , size = np.shape(y_train))/np.sqrt(np.prod(np.shape(y_train)[1:]))
    n_test = np.random.normal(loc = 0 , scale = noise_sigma_test , size = np.shape(y_test))/np.sqrt(np.prod(np.shape(y_test)[1:]))
    
    y_train = y_train + n_train
    y_test = y_test + n_test
    
    
    xp_train = np.matmul(y_train, A_pinv.T)    
    xp_test = np.matmul(y_test, A_pinv.T)
    
    xp_train = np.reshape(xp_train , [b_train , image_size,image_size,c])
    xp_test = np.reshape(xp_test , [b_test , image_size,image_size,c])
    

    return xp_train , xp_test
    


def denoising_op(x_train , x_test, noise_snr = 100):
    '''Denoising forward operator'''
    
    b_train = np.shape(x_train)[0]
    b_test = np.shape(x_test)[0]
    
    noise_sigma_train = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
        np.square(np.reshape(x_train, (b_train , -1))) , -1)))
    
   
    
    noise_sigma_test = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
        np.square(np.reshape(x_test, (b_test , -1))) , -1)))
    

    
    n_train = np.random.normal(loc = 0 , scale = noise_sigma_train , size = np.shape(x_train))/np.sqrt(np.prod(np.shape(x_train)[1:]))
    n_test = np.random.normal(loc = 0 , scale = noise_sigma_test , size = np.shape(x_test))/np.sqrt(np.prod(np.shape(x_test)[1:]))
    
    y_train = x_train + n_train
    y_test = x_test + n_test
            
    return y_train , y_test



def super_resolution_op(x_train , x_test, down_factor = 4 , noise_snr = 100):
    '''Super resolution forward operator'''
    b_train, h, _, ch = np.shape(x_train)
    b_test = np.shape(x_test)[0]
    
    r = h//down_factor
    
    
    y_train = image_resizer(image_resizer(x_train, r) , h)
    y_test = image_resizer(image_resizer(x_test, r) , h)

    
    noise_sigma_train = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
        np.square(np.reshape(y_train, (b_train , -1))) , -1)))
    
    noise_sigma_test = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
        np.square(np.reshape(y_test, (b_test , -1))) , -1)))
    
    n_train = np.random.normal(loc = 0 , scale = noise_sigma_train , size = np.shape(y_train))/np.sqrt(np.prod(np.shape(y_train)[1:]))
    n_test = np.random.normal(loc = 0 , scale = noise_sigma_test , size = np.shape(y_test))/np.sqrt(np.prod(np.shape(y_test)[1:]))
    
    y_train = y_train + n_train
    y_test = y_test + n_test
    

    return y_train , y_test
    


def random_mask_op(x_train , x_test, prob_to_keep = 0.1 , noise_snr = 100):
    '''Random mask forward operator'''
    mask_train = (np.random.uniform(size = np.shape(x_train)) < prob_to_keep).astype(np.float32)
    y_train = x_train*mask_train
    
    mask_test = (np.random.uniform(size= np.shape(x_test)) < prob_to_keep).astype(np.float32)
    y_test = x_test*mask_test
    
    b_train = np.shape(x_train)[0]
    b_test = np.shape(x_test)[0]
    
    noise_sigma_train = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
        np.square(np.reshape(y_train, (b_train , -1))) , -1)))
    
    noise_sigma_test = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
        np.square(np.reshape(y_test, (b_test , -1))) , -1)))
    
    n_train = np.random.normal(loc = 0 , scale = noise_sigma_train , size = np.shape(y_train))/np.sqrt(np.prod(np.shape(y_train)[1:]))
    n_test = np.random.normal(loc = 0 , scale = noise_sigma_test , size = np.shape(y_test))/np.sqrt(np.prod(np.shape(y_test)[1:]))
    
    
    y_train = y_train + n_train
    y_test = y_test + n_test
    
    return y_train , y_test
    
 
    
def mask_op(x_train , x_test, mask_size = 32 , noise_snr = 100):
    '''Mask forward operator with random locations'''
    b_train, h, w, c = np.shape(x_train)
    k = 1
    
    y_train = np.zeros([b_train * k, h, w, c])
    x_train_new = np.zeros([b_train * k, h, w, c])
    for i in range(b_train):
        
        for j in range(k):
        
            c_h = np.random.randint(mask_size//2, h - mask_size//2)
            c_w = np.random.randint(mask_size//2, w - mask_size//2)
            
            sh = c_h - mask_size//2
            eh = c_h + mask_size//2 
            
            sw = c_w - mask_size//2
            ew = c_w + mask_size//2 
        
            yt = x_train[i,:,:,:]
            yt[sh:eh, sw:ew,:] = 0.0
            
            b_train = np.shape(x_train)[0]
            
            noise_sigma_train = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
                np.square(np.reshape(yt, (1 , -1))) , -1)))
            
            n_train = np.random.normal(loc = 0 , scale = noise_sigma_train,
                                       size = [h,w,c])/np.sqrt(np.prod([h,w,c]))
            y_train[i * k + j,:,:,:] = yt + n_train
            x_train_new[i * k + j,:,:,:] = x_train[i,:,:,:]
            
    
    
    b_test, h, w, c = np.shape(x_test)
    y_test = np.zeros([b_test, h, w, c])
    for i in range(b_test):
        
        
        c_h = np.random.randint(mask_size//2, h - mask_size//2)
        c_w = np.random.randint(mask_size//2, w - mask_size//2)
        
        sh = c_h - mask_size//2
        eh = c_h + mask_size//2 
        
        sw = c_w - mask_size//2
        ew = c_w + mask_size//2 
        
        yt = x_test[i,:,:,:]
        yt[sh:eh, sw:ew,:] = 0.0
        
        b_test = np.shape(x_test)[0]
        
        noise_sigma_test = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
                np.square(np.reshape(yt, (1 , -1))) , -1)))
        
        n_test = np.random.normal(loc = 0 , scale = noise_sigma_test ,
                                   size = [h,w,c])/np.sqrt(np.prod([h,w,c]))
        
        y_test[i,:,:,:] = yt + n_test
        
    return y_train , y_test



def Dataset_preprocessing(dataset = 'MNIST', batch_size = 64 ,
                          task='sr' , noise_snr = 100 ,
                          problem_factor = 4,
                          resolution = 64,
                          missing_cone = 'vertical',
                          epsilon_r = 6,
                          setup = 'full',
                          conditions = 'es'):
    
    if dataset == 'mnist':

        (training_images, y_train), (testing_images, y_test) = tf.keras.datasets.mnist.load_data()
        
        training_images = np.expand_dims(training_images, axis = 3)
        testing_images = np.expand_dims(testing_images, axis = 3)
        
        r = 32 # image resolution
        pipeline = False
        
        
    elif dataset == 'voxceleb':

        y = np.load('datasets/voxceleb/cropped_y.npy')
        images = np.load('datasets/voxceleb/cropped_x.npy')
        
        training_images, testing_images, y_train, y_test = train_test_split(images,
                                                            y,
                                                            test_size=0.05,
                                                            random_state=42)
        
        
        r = 64
        pipeline = False
        


    elif dataset == 'celeba':
        celeba = np.load('datasets/celeba/celeba_64_100k.npy')
        # celeba = shuffle(celeba)
        training_images, testing_images = np.split(celeba, [80000], axis=0)
        r = 64 # image resolution
        
        pipeline = False
        
        
    elif dataset == 'limited-CT':
        
        r = resolution
        
        if r == 256:
            
            pipeline = True
            
            x_folder = 'datasets/limited-CT/images/gt_train'
            
            if missing_cone == 'vertical':
                y_folder = 'datasets/limited-CT/images/fbp_train_vertical_snr_40'
                
            elif missing_cone == 'horizontal':
                y_folder = 'datasets/limited-CT/images/fbp_train_horizontal_snr_40'


            training_images = tf.keras.preprocessing.image_dataset_from_directory(x_folder,
                                                        label_mode=None,
                                                        batch_size=batch_size,
                                                        image_size=(r,r),
                                                        color_mode='grayscale',
                                                        shuffle=False,
                                                        seed = 0,
                                                        validation_split = 0.01,
                                                        subset = 'training')
    
            testing_images = tf.keras.preprocessing.image_dataset_from_directory(x_folder,
                                                        label_mode=None,
                                                        batch_size=batch_size,
                                                        image_size=(r,r),
                                                        color_mode='grayscale',
                                                        shuffle=False,
                                                        seed = 0,
                                                        validation_split = 0.01,
                                                        subset = 'validation')
            
            y_train = tf.keras.preprocessing.image_dataset_from_directory(y_folder,
                                                        label_mode=None,
                                                        batch_size=batch_size,
                                                        image_size=(r,r),
                                                        color_mode='grayscale',
                                                        shuffle=False,
                                                        seed = 0,
                                                        validation_split = 0.01,
                                                        subset = 'training')
    
            y_test = tf.keras.preprocessing.image_dataset_from_directory(y_folder,
                                                        label_mode=None,
                                                        batch_size=batch_size,
                                                        image_size=(r,r),
                                                        color_mode='grayscale',
                                                        shuffle=False,
                                                        seed = 0,
                                                        validation_split = 0.01,
                                                        subset = 'validation')
            
        
        elif r== 64:
            
            if missing_cone == 'vertical':
                data = np.load('datasets/limited-CT/vertical_snr25.0.npz')
            elif missing_cone == 'horizontal':
                data = np.load('datasets/limited-CT/horizontal_snr25.0.npz')

            y_train = data['y_train']
            y_test = data['y_test']
            training_images = data['x_train']
            testing_images = data['x_test']
            
            
            pipeline = False

        
        
        
    elif dataset == 'scattering': # Inverse Scattering
        
        # Target signals (random ellipses)
        x = np.load('datasets/scattering/ellipses_diverse_64.npy')
        training_images = x[:55000]
        testing_images = x[55000:]
    
        file_name = 'ellipses_'+str(epsilon_r)+ '_' + str(setup) +'_64_snr30.0_d2.0.npz'
        data = np.load('datasets/scattering/' + file_name)
        
        if conditions == 'bp':
            # Backprojection conditions
            y_train = data['bp_train']
            y_test = data['bp_test']

        
        elif conditions == 'es':
            # Scattered fields conditions
            y_train = data['Es_train']
            y_test = data['Es_test']
            
            n_measure = np.shape(y_train)[1]
            y_train = y_train[:,::n_measure//36]
            y_test = y_test[:,::n_measure//36]
            
            removed_list = [0,9,18,27]
            selected_list = np.arange(36)
            selected_list = np.delete(selected_list, removed_list).tolist()
            y_train = np.repeat(y_train[:,selected_list][:,:,selected_list] , 2 ,  axis = 3)
            y_test = np.repeat(y_test[:,selected_list][:,:,selected_list] , 2 ,  axis = 3)

        
        
        r = 64 # image resolution
        pipeline = False
        
        
        

        
        
        
    elif dataset == 'random_square':
        
        N = 60000
        r = 64
        s = 8
        K = 8
        samples = np.zeros([N,r,r,1])
        for n in range(N):
            
            for k in range(K):
                
                loc = np.random.randint(s//2, high= r - s//2, size=[2], dtype=int)
                samples[n , loc[0] - s//2: loc[0] + s//2 , loc[1] - s//2: loc[1] + s//2] = 1
                
        training_images, testing_images = train_test_split(samples,
                                                     test_size=0.05,
                                                     random_state=42)
        pipeline = False    
        
        


    if pipeline == False:
        
        training_images = image_resizer(training_images, r)
        testing_images = image_resizer(testing_images, r)
        
        training_images = data_normalization(training_images)
        testing_images = data_normalization(testing_images)
        

    if task == 'cs' : # Class-based image generation
        
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        
    elif task == 'denoising':
        
        y_train , y_test = denoising_op(training_images + 1.0,
                                        testing_images + 1.0,
                                        noise_snr)
        y_train -= 1.0
        y_test -= 1.0
        
    
    elif task == 'sr': # Super Resolution
    
        
        y_train , y_test = super_resolution_op(training_images+ 1.0,
                                               testing_images+ 1.0,
                                               int(problem_factor),
                                               noise_snr)
        y_train -= 1.0
        y_test -= 1.0
            
    elif task == 'random_mask':
        
        y_train , y_test = random_mask_op(training_images+ 1.0,
                                          testing_images+ 1.0,
                                          problem_factor,
                                          noise_snr)
        y_train -= 1.0
        y_test -= 1.0
        
        
    elif task == 'mask':
        
        y_train , y_test = mask_op(training_images+ 1.0,
                                          testing_images+ 1.0,
                                          int(problem_factor),
                                          noise_snr)
        y_train -= 1.0
        y_test -= 1.0
            
        
    elif task == 'scattering':
        
        y_train = y_train
        y_test = y_test
        

        
    elif task == 'traveltime':
        
        y_train , y_test = traveltime_op(training_images+ 1.0,
                                          testing_images+ 1.0,
                                          noise_snr)
        y_train -= 1.0
        y_test -= 1.0


    elif task == 'limited-CT':
        
        if pipeline:
            y_train = y_train
            y_test = y_test
            
        else:
            y_train = data_normalization(y_train)
            y_test = data_normalization(y_test)

    else:
        
        raise Exception('Task is not valid')
    
    
    
    if pipeline == False:

        training_images = tf.convert_to_tensor(training_images, tf.float32)
        testing_images = tf.convert_to_tensor(testing_images, tf.float32)
        y_train = tf.convert_to_tensor(y_train, tf.float32)
        y_test = tf.convert_to_tensor(y_test, tf.float32)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((training_images, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((testing_images, y_test))
    
        train_dataset = train_dataset.shuffle(batch_size * 3).batch(batch_size).prefetch(3 * batch_size)
        test_dataset = test_dataset.batch(np.shape(testing_images)[0])

    else:
                
        max_pix_image = np.max(next(iter(training_images)))/2
        max_pix_y = np.max(next(iter(y_train)))/2
        
        norm_layer_image = tf.keras.layers.experimental.preprocessing.Rescaling(1./max_pix_image,
                                                                 offset=-1)
        
        norm_layer_y = tf.keras.layers.experimental.preprocessing.Rescaling(1./max_pix_y,
                                                                 offset=-1)
        
        training_images = training_images.map(lambda x : norm_layer_image(x))
        testing_images = testing_images.map(lambda x : norm_layer_image(x))
        
        y_train = y_train.map(lambda x : norm_layer_y(x))
        y_test = y_test.map(lambda x : norm_layer_y(x))
        
   
        train_dataset = tf.data.Dataset.zip((training_images , y_train)).prefetch(3*batch_size)
        test_dataset = tf.data.Dataset.zip((testing_images , y_test)) 

    return train_dataset , test_dataset
      
        
 
def flags():

    parser = argparse.ArgumentParser()
     
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=300,
        help='number of epochs to train for')
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='batch_size')

   
    parser.add_argument(
        '--dataset', 
        type=str,
        default='celeba',
        help='which dataset to work with')
    
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate')
    
    
    parser.add_argument(
        '--ml_threshold', 
        type=int,
        default= 150,
        help='when should ml training begin')


    parser.add_argument(
        '--model_depth',
        type=int,
        default= 3,
        help='revnet depth of injective subnetwork')
    
    parser.add_argument(
        '--latent_depth',
        type=int,
        default= 4,
        help='revnet depth of bijective subnetwork')
    
    
    parser.add_argument(
        '--learntop',
        type=int,
        default=1,
        help='trainable mu and sigma of the bese Gaussian')
    
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=0,
        help='GPU number')

    parser.add_argument(
        '--remove_all',
        type= int,
        default= 1,
        help='Remove the previous experiment if exists')
    
    parser.add_argument(
        '--desc',
        type=str,
        default='Default',
        help='add a small descriptor to the experiment folder name')
    
    
    parser.add_argument(
        '--problem',
        type=str,
        default= 'sr',
        help='Which inverse problem')
    
    parser.add_argument(
        '--noise_snr',
        type=float,
        default= 100,
        help='SNR of the additive noise')
    
    parser.add_argument(
        '--problem_factor',
        type=float,
        default= 4,
        help='specific factor of the inverse problem')
    
    parser.add_argument(
        '--resolution',
        type=int,
        default= 64,
        help='Resolution of dataset just for limited-CT problem (64 or 256)')
    
    parser.add_argument(
        '--missing_cone',
        type=str,
        default= 'vertical',
        help='The missing cone of limited-CT  problem(vertical or horizontal)')
    
    
    parser.add_argument(
        '--epsilon_r',
        type=float,
        default= 6,
        help='epsilon_r for the scattering problem (1.5, 2 or 6)')
    
    parser.add_argument(
        '--setup',
        type=str,
        default= 'full',
        help='The incident waves and receivers setup for the scattering problem (full or slice)')
    

    parser.add_argument(
        '--conditions',
        type=str,
        default= 'es',
        help='Type of conditioning data (scattered_fields or Backprojections) for scattering problem (es or bp)')
    
    parser.add_argument(
        '--train',
        type=int,
        default= 1,
        help='Train the model or just test')
    
    

    
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed
