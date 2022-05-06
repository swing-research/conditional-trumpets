from models import generator, latent_generator
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
from utils import *
import os
import shutil
from time import time


tfb = tfp.bijectors
tfd = tfp.distributions

FLAGS, unparsed = flags()

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
dataset = FLAGS.dataset
lr = FLAGS.lr
gpu_num = FLAGS.gpu_num
learntop = bool(FLAGS.learntop)
remove_all = bool(FLAGS.remove_all)
desc = FLAGS.desc
ml_threshold = FLAGS.ml_threshold
model_depth = FLAGS.model_depth
latent_depth = FLAGS.latent_depth
problem = FLAGS.problem
noise_snr = FLAGS.noise_snr
problem_factor = FLAGS.problem_factor
run_train = bool(FLAGS.train)
missing_cone = FLAGS.missing_cone
resolution = FLAGS.resolution
setup = FLAGS.setup
epsilon_r = FLAGS.epsilon_r
conditions = FLAGS.conditions


all_experiments = 'experiments/' # experiments folder
if os.path.exists(all_experiments) == False:

    os.mkdir(all_experiments)

# experiment path
exp_path = all_experiments + 'Injective_' + \
    dataset + '_' + 'model_depth_%d' % (model_depth,) + '_' + 'latent_depth_%d'% (latent_depth,) + '_learntop_%d' \
        % (int(learntop)) + '_' + desc


if os.path.exists(exp_path) == True and remove_all == True:
    shutil.rmtree(exp_path) # Remove the previous experiment if exists

if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the selected GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)



class Prior(layers.Layer):
    """Defines the low dimensional distribution as Guassian"""
    def __init__(self, **kwargs):
        super(Prior, self).__init__()
        
        latent_dim = kwargs.get('latent_dim', 64)
            
        self.mu = tf.Variable(tf.zeros(latent_dim),
                              dtype=tf.float32, trainable=learntop)
        self.logsigma = tf.Variable(tf.zeros(latent_dim),
                                    dtype=tf.float32, trainable=learntop)

        self.prior = tfd.MultivariateNormalDiag(
            self.mu, tf.math.exp(self.logsigma))




def train(num_epochs,
          batch_size,
          dataset,
          lr,
          exp_path,
          problem,
          noise_snr,
          problem_factor):


    # Experiment setup:
    print('Experiment setup:')
    print('---> num_epochs: {}'.format(num_epochs))
    print('---> batch_size: {}'.format(batch_size))
    print('---> dataset: {}'.format(dataset))
    print('---> Learning rate: {}'.format(lr))
    print('---> experiment path: {}'.format(exp_path))
    print('---> problem: {}'.format(problem))
    print('---> noise_snr: {}'.format(noise_snr))
    print('---> problem_factor: {}'.format(problem_factor))
    
    
    # Logs
    if os.path.exists(os.path.join(exp_path, 'logs')):
        shutil.rmtree(os.path.join(exp_path, 'logs'))
    
    # MSE loss of training data
    MSE_train_log_dir = os.path.join(exp_path, 'logs', 'MSE_train')
    MSE_train_summary_writer = tf.summary.create_file_writer(MSE_train_log_dir)
    MSE_train_loss_metric = tf.keras.metrics.Mean(
        'MSE_train_loss', dtype=tf.float32)

    # MSE loss of test data
    MSE_test_log_dir = os.path.join(exp_path, 'logs', 'MSE_test')
    MSE_test_summary_writer = tf.summary.create_file_writer(MSE_test_log_dir)
    MSE_test_loss_metric = tf.keras.metrics.Mean('MSE_test_loss', dtype=tf.float32)

    # -log-likelihood loss of training data
    ML_log_dir = os.path.join(exp_path, 'logs', 'ML')
    ML_summary_writer = tf.summary.create_file_writer(ML_log_dir)
    ML_loss_metric = tf.keras.metrics.Mean('ML_loss', dtype=tf.float32)
    
    # -Log p(z) in ML loss
    pz_log_dir = os.path.join(exp_path, 'logs', 'pz')
    pz_summary_writer = tf.summary.create_file_writer(pz_log_dir)
    pz_metric = tf.keras.metrics.Mean(
        'pz', dtype=tf.float32)
    
    # -Log det (J) in ML loss
    jacobian_log_dir = os.path.join(exp_path, 'logs', 'jacobian')
    jacobian_summary_writer = tf.summary.create_file_writer(jacobian_log_dir)
    jacobian_metric = tf.keras.metrics.Mean(
        'jacobian', dtype=tf.float32)
    
    # SNR of the MMSE estimate
    snr_log_dir = os.path.join(exp_path, 'logs', 'snr')
    snr_summary_writer = tf.summary.create_file_writer(snr_log_dir)
    snr_metric = tf.keras.metrics.Mean(
        'snr', dtype=tf.float32)
    
    
    # SSIM of the MMSE estimate
    ssim_log_dir = os.path.join(exp_path, 'logs', 'ssim')
    ssim_summary_writer = tf.summary.create_file_writer(ssim_log_dir)
    ssim_metric = tf.keras.metrics.Mean(
        'ssim', dtype=tf.float32)
    
    
    # Data loader
    train_dataset, test_dataset = Dataset_preprocessing(dataset=dataset,
                                                        batch_size = batch_size,
                                                        task = problem,
                                                        noise_snr = noise_snr,
                                                        problem_factor = problem_factor,
                                                        resolution = resolution,
                                                        missing_cone = missing_cone,
                                                        epsilon_r = epsilon_r,
                                                        setup = setup,
                                                        conditions = conditions)
    
    
    print('Dataset is loaded: training and test dataset shape: {} {}'.
          format(np.shape(next(iter(train_dataset))[0]), np.shape(next(iter(test_dataset))[0])))
    
    print('training and test measurments: {} {}'.
          format(np.shape(next(iter(train_dataset))[1]), np.shape(next(iter(test_dataset))[1])))

    _ , image_size , _ , c = np.shape(next(iter(train_dataset))[0])
    y_shape = np.shape(next(iter(train_dataset))[1])
    # f will be used in model to select the appropraite
    # architecture best on the image resolution
    f = image_size//64 
    f = 1 if f < 1 else f # For image size 32
    
    latent_dim = 4*f *4*f *4*c # dimension of the latent space

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # Optimizer of injective network
    f_optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # Optimizer of bijective network

    pz = Prior(latent_dim = latent_dim)
    
    model = generator(revnet_depth = model_depth,
                      f = f,
                      c = c,
                      image_size = image_size) # Injective subnetwork
    latent_model = latent_generator(revnet_depth = latent_depth,
                                    f = f,
                                    c = c,
                                    image_size = image_size) # Bijective subnetwork

     # call generator once to set weights (Data-dependent initialization)
    dummy_x , dummy_y = next(iter(train_dataset))
    dummy_z, _ = model(dummy_x, dummy_y, reverse=False)
    dummy_l_z , _ = latent_model(dummy_z, dummy_y,  reverse=False)
    
    
    # Checkpoints
    ckpt = tf.train.Checkpoint(pz = pz , model=model,optimizer=optimizer,
        latent_model=latent_model,f_optimizer=f_optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=5)

    ckpt.restore(manager.latest_checkpoint)
    
    @tf.function
    def train_step_mse(sample , y):
        """MSE training of the injective subnetwork"""

        with tf.GradientTape() as tape:
            
            MSE = tf.keras.losses.MeanSquaredError()
            
            z , _ = model(sample, y, reverse= False)
            recon = model(z, y, reverse = True)[0]
            
            mse_loss = MSE(sample , recon)
            loss = mse_loss 
            variables= tape.watched_variables()
            
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

        return loss
    
    
    
    @tf.function
    def train_step_ml(sample , y):
        """ML training of the bijective network"""
        
        with tf.GradientTape() as tape:
            
            latent_sample, obj = latent_model(sample, y, reverse=False)
            p = -tf.reduce_mean(pz.prior.log_prob(latent_sample))
            j = -tf.reduce_mean(obj) # Log-det of Jacobian
            loss =  p + j
            variables= tape.watched_variables()
            
            grads = tape.gradient(loss, variables)
            f_optimizer.apply_gradients(zip(grads, variables))

        return loss , p , j

   

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    # Train the models
    if run_train:
    
        counter = 0
        z_inters = np.zeros([len(list(train_dataset)) * batch_size ,
                             latent_dim]) # Latent codes of the training data in the intermediate space
        y_inters = np.zeros([len(list(train_dataset))* batch_size] + y_shape[1:])
        for epoch in range(num_epochs):
            
            epoch_start = time()
            
            if epoch < ml_threshold:
                for x,y in train_dataset:
                    # MSE traiing of the injective subnetwork for ml-threshold number of epochs
                    _ = train_step_mse(x, y)
                    ml_loss = 0
                    p = 0
                    j = 0

            elif epoch == ml_threshold:
                # Compute the latent codes of the training data in
                # the intermediate space and save them
                if ml_threshold == 0:
                    ml_loss = 0
                    p = 0
                    j = 0
                counter = 0
                for x,y in train_dataset:
                    # MSE traiing of the injective network for ml-threshold epochs
                
                    z_inter, _ = model(x, y, reverse= False)
                    if tf.shape(z_inter)[0] == batch_size:
                        z_inters[counter*batch_size:(counter+1)*batch_size] = z_inter.numpy()
                        y_inters[counter*batch_size:(counter+1)*batch_size] = y.numpy()
                        
                        counter = counter + 1
                    
                
                z_inters = tf.convert_to_tensor(z_inters, tf.float32)
                y_inters = tf.convert_to_tensor(y_inters, tf.float32)
                
                z_inters_dataset = tf.data.Dataset.from_tensor_slices((z_inters, y_inters))
                z_inters_dataset = z_inters_dataset.shuffle(batch_size * 3).batch(batch_size , drop_remainder = True).prefetch(5)
                
            else:
                # ML training of the bijective subnetwork after ml threshold epochs of MSE training
                
                for x,y in z_inters_dataset:
                    
                    ml_loss , p , j = train_step_ml(x , y)
            
            if epoch == 0:
                
                # Just for the first iteration of the first epoch
                # to calculate the number of trainable parametrs
                x , y  = next(iter(train_dataset))
                with tf.GradientTape() as tape:
                    
                    z_batch1 , _ = model(x, y, reverse= False)
                    variables_model = tape.watched_variables()
                
                with tf.GradientTape() as tape:
                    
                    _, _ = latent_model(z_batch1, y, reverse=False)
                    variables_latent_model = tape.watched_variables()
                    
                parameters_model = np.sum([np.prod(v.get_shape().as_list()) for v in variables_model])
                parameters_latent_model = np.sum([np.prod(v.get_shape().as_list()) for v in variables_latent_model])
                print('Number of trainable_parameters of injective model: {}'.format(parameters_model))
                print('Number of trainable_parameters of bijective model: {}'.format(parameters_latent_model))
                print('Total number of trainable_parameters: {}'.format(parameters_model + parameters_latent_model))
                
            
            if epoch % 1 == 0:
                
                ML_loss_metric.update_state(ml_loss)
                pz_metric.update_state(p)
                jacobian_metric.update_state(j)
                
                sample_number = 25 # Number of samples to show
                
                training_images , y_train  = next(iter(train_dataset))
                testing_images , y_test = next(iter(test_dataset))

                z_hat_train = model(training_images[:sample_number], y_train[:sample_number],
                                   reverse= False)[0] 
                # Latent codes of the intermediate space of training images
                x_hat_train = model(z_hat_train , y_train[:sample_number] , reverse = True)[0]
                # Reconstrcted training images (Projection on the manifold)
                
                train_relative_mse = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.square(training_images[:sample_number] - x_hat_train) ,
                                                           axis = [1,2,3]))/tf.math.sqrt(tf.reduce_sum(tf.square(training_images[:sample_number]) , axis = [1,2,3])))
                
                # Relative MSE of reconstructions of training data
                MSE_train_loss_metric.update_state(train_relative_mse)
                
                
                z_hat_test = model(testing_images[:sample_number], y_test[:sample_number],
                                   reverse= False)[0] 
                # Latent codes of the intermediate space of testing images
                x_hat_test = model(z_hat_test , y_test[:sample_number] , reverse = True)[0]
                # Reconstrcted testing images (Projection on the manifold)
                
                test_mse = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.square(testing_images[:sample_number] - x_hat_test) ,
                                                           axis = [1,2,3]))/tf.math.sqrt(tf.reduce_sum(tf.square(testing_images[:sample_number]) , axis = [1,2,3])))
                # Relative MSE of reconstrcutions of test data
                MSE_test_loss_metric.update_state(test_mse)
        
    
    
                # Conditional_sampling from posterior distribution
                if problem == 'cs':
                    num_classes = np.shape(y_test)[1]
                    x_sampled_conditional = classified_sampling(pz, model, latent_model , num_classes)
                
                else:
                    n_test = 5 # Number of test samples
                    n_sample_show = 4 # Number of posterior samples to show for each test sample
                    n_average = 25 # number of posterior samples used for MMSE and UQ estimation
                    x_sampled_conditional , y_test_selected , snr, SSIM = conditional_sampling(pz , model ,
                                                                                               latent_model ,
                                                                                               testing_images ,
                                                                                               y_test , n_average, n_test,
                                                                                               n_sample_show)
            
                    snr_metric.update_state(snr)
                    ssim_metric.update_state(SSIM)
                    

                # Saving experiment results
                samples_folder = os.path.join(exp_path, 'Generated_samples')
                if not os.path.exists(samples_folder):
                    os.mkdir(samples_folder)
                image_path_inverse_test = os.path.join(
                    samples_folder, 'reconstruction_test') # Projected test samples on the manifold
        
                if not os.path.exists(image_path_inverse_test):
                    os.mkdir(image_path_inverse_test)
        
                ngrid = int(np.sqrt(sample_number))
        
                cv2.imwrite(os.path.join(image_path_inverse_test, 'recon_epoch %d.png' % (epoch,)),
                            x_hat_test.numpy()[:, :, :, ::-1].reshape(
                    ngrid, ngrid,
                    image_size, image_size, c).swapaxes(1, 2)
                    .reshape(ngrid*image_size, -1, c)*127.5 + 127.5) # Reconstructed test images
                
                
                cv2.imwrite(os.path.join(image_path_inverse_test, 'gt_epoch %d.png' % (epoch,)),
                            testing_images.numpy()[:sample_number, :, :, ::-1].reshape(
                    ngrid, ngrid,
                    image_size, image_size, c).swapaxes(1, 2)
                    .reshape(ngrid*image_size, -1, c)* 127.5 + 127.5) # Ground truth test images
                
                image_path_sampled = os.path.join(samples_folder, 'posterior samples')
                if os.path.exists(image_path_sampled) == False:
                    os.mkdir(image_path_sampled)
        

                if problem == 'cs':
                    
                    cv2.imwrite(os.path.join(image_path_sampled, 'conditional_samples_epoch %d.png' % (epoch,)),
                                x_sampled_conditional[:, :, :, ::-1].reshape(
                        num_classes, 5,
                        image_size, image_size, c).swapaxes(1, 2)
                        .reshape(num_classes*image_size, -1, c)*127.5 + 127.5) # samples from posterior distribution
                    
                    
                else:

                    cv2.imwrite(os.path.join(image_path_sampled, 'conditional_sampled_epoch %d.png' % (epoch,)),
                                x_sampled_conditional[:, :, :, ::-1].reshape(
                        n_test, n_sample_show + 5,
                        image_size, image_size, c).swapaxes(1, 2)
                        .reshape(n_test*image_size, -1, c)*127.5 + 127.5) # samples from posterior distribution
                    
                    
                
                # Saving logs
                with MSE_train_summary_writer.as_default():
                    tf.summary.scalar(
                        'MSE_train', MSE_train_loss_metric.result(), step=epoch)
        
                with MSE_test_summary_writer.as_default():
                    tf.summary.scalar(
                        'MSE_test', MSE_test_loss_metric.result(), step=epoch)
        
                with ML_summary_writer.as_default():
                    tf.summary.scalar(
                        'ML_loss', ML_loss_metric.result(), step=epoch)
        
                
                with pz_summary_writer.as_default():
                    tf.summary.scalar(
                        'pz', pz_metric.result(), step=epoch)
                    
                
                with jacobian_summary_writer.as_default():
                    tf.summary.scalar(
                        'jacobian', jacobian_metric.result(), step=epoch)
                    
                    
                with snr_summary_writer.as_default():
                    tf.summary.scalar(
                        'snr', snr_metric.result(), step=epoch)
                    
                    
                with ssim_summary_writer.as_default():
                    tf.summary.scalar(
                        'ssim', ssim_metric.result(), step=epoch)
                 
                    
                
                save_path = manager.save()
                print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
                
                epoch_end = time()
                time_ell = epoch_end - epoch_start
                print("Epoch {:03d}: MSE train: {:.3f} / MSE test: {:.3f} / ML Loss: {:.3f} / epoch time:{:.3f} "
                      .format(epoch, MSE_train_loss_metric.result().numpy(), MSE_test_loss_metric.result().numpy(),
                              ML_loss_metric.result().numpy(), time_ell))
                
                MSE_train_loss_metric.reset_states()
                MSE_test_loss_metric.reset_states()
                ML_loss_metric.reset_states()
                pz_metric.reset_states()
                jacobian_metric.reset_states()
                snr_metric.reset_states()
                ssim_metric.reset_states()
                
    
    if not problem == 'cs':
        # Show more posterior samples samples after training
        n_test = 5 # Number of test samples
        n_sample_show = 20 # Number of posterior samples to show for each test sample
        n_average = 25 # number of posterior samples used for MMSE and UQ estimation
        testing_images , y_test = next(iter(test_dataset))      
        
        x_sampled_conditional = conditional_sampling(pz , model ,
                                                    latent_model ,
                                                    testing_images ,
                                                    y_test , n_average, n_test,
                                                    n_sample_show)[0]
        
        cv2.imwrite(os.path.join(exp_path, 'posterior_samples.png'),
                    x_sampled_conditional[:, :, :, ::-1].reshape(
            n_test, n_sample_show + 5,
            image_size, image_size, c).swapaxes(1, 2)
            .reshape(n_test*image_size, -1, c)*127.5 + 127.5)
                



if __name__ == '__main__':
    train(num_epochs,
          batch_size,
          dataset,
          lr,
          exp_path,
          problem,
          noise_snr,
          problem_factor)
