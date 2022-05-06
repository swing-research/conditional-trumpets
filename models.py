import tensorflow as tf
import numpy as np
import glow_ops as g
from time import time


class generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(generator, self).__init__()
        """Injective subnetwork
        """
        self.depth = kwargs.get('revnet_depth', 1) # revnet depth
        self.f = kwargs.get('f', 1)
        self.c = kwargs.get('c', 3)
        self.image_size = kwargs.get('image_size', 32)
        
        self.squeeze = g.upsqueeze(factor=2)
        self.revnets = [g.revnet(depth= self.depth , latent_model = False) 
                        for _ in range(6)] # Bijective revnets
        
        self.inj_rev_steps = [g.revnet_step(layer_type='injective', latent_model = False) 
                              for _ in range(6)]
        

    def call(self, x , y, reverse=False):
        

        if reverse:
            x = tf.reshape(x, [-1,4*self.f, 4*self.f, 4*self.c])
        ops = [
        self.squeeze,
        self.revnets[0],
        self.inj_rev_steps[0],
        self.squeeze,
        self.revnets[1],
        self.inj_rev_steps[1],
        self.squeeze,
        self.revnets[2],
        self.inj_rev_steps[2],
        self.revnets[3],
        self.inj_rev_steps[3],
        ]
        
        if self.image_size >= 64:
            
            ops += [self.inj_rev_steps[4],
            self.revnets[4],
            self.squeeze,
            self.inj_rev_steps[5],
            self.revnets[5]
            ]

        if reverse:
            ops = ops[::-1]

        objective = 0.0
        for op in ops:
            
            if type(op) == g.upsqueeze:
                x, curr_obj = op(x, reverse= reverse)
            else:
                x, curr_obj = op(x, y, reverse= reverse)

            objective += curr_obj
        
        
        if not reverse:
 
            x = tf.reshape(x, (-1, 4*self.f *4*self.f *4*self.c))

        return x, objective
 


class latent_generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(latent_generator, self).__init__()
        """ Bijective subnetwork"""
        self.depth = kwargs.get('revnet_depth', 1) # revnet depth
        self.f = kwargs.get('f', 1)
        self.c = kwargs.get('c', 3)
        self.image_size = kwargs.get('image_size', 32)
        self.squeeze = g.upsqueeze(factor=2)
        
        self.revnets = [g.revnet(depth = self.depth , latent_model = True) 
        for _ in range(6)]
    def call(self, x, y, reverse=False):

        if self.image_size == 256:
            if not reverse:
                x = tf.reshape(x, [-1,4*self.f,4*self.f, 4 *  self.c])
            else:
                x = tf.reshape(x, [-1,self.f,self.f, 4 * 4 * 4 *  self.c])
                
            ops = [
            self.revnets[0],
            self.revnets[1],
            self.squeeze,
            self.revnets[2],
            self.revnets[3],
            self.squeeze,
            self.revnets[4],
            self.revnets[5]]
            
            
        else:
            x = tf.reshape(x, [-1,4*self.f, 4*self.f, 4*self.c])
        
            ops = [
            self.revnets[0],
            self.revnets[1],
            self.revnets[2],
            self.revnets[3],
            self.revnets[4],
            self.revnets[5]
            ]
            

        if reverse:
            ops = ops[::-1]

        objective = 0.0

        for op in ops:
            
            if type(op) == g.upsqueeze:
                x, curr_obj = op(x, reverse= reverse)
            else:
                x, curr_obj = op(x, y, reverse= reverse)
                
            objective += curr_obj
        

        x = tf.reshape(x, (-1, 4*self.f *4*self.f *4*self.c))

        return x, objective
 


def unit_test_generator():
    
    MSE = tf.keras.losses.MeanSquaredError()
    
    x = tf.random.normal(shape = [100,64,64,3])
    z = tf.random.normal(shape = [100,4*4*12])
    y = tf.random.normal(shape = [100,64,64,3])
    
    g = generator(f = 1, c = 3 , image_size = 64)
    x_int , _ = g(x , y , reverse = False)
    print(np.shape(x_int))
    z_int , _ = g(z , y , reverse = True)
    print(np.shape(z_int))
    z_hat , _ = g(z_int , y , reverse = False)
    
    loss = MSE(z , z_hat)
    
    print(loss)
    

def unit_test_latent_generator():
    
    start = time()
    
    MSE = tf.keras.losses.MeanSquaredError()
    
    x = tf.random.normal(shape = [10 , 4,4,4])
    z = tf.random.normal(shape = [10 , 4*4*4])
    y = tf.random.normal(shape = [10 , 64,64,1])
    
    g = latent_generator(f = 1 , c = 1 , image_size = 64)
    x_int , _ = g(x , y , reverse = False)
    print(np.shape(x_int))
    z_int , _ = g(z , y , reverse = True)
    print(np.shape(z_int))
    z_hat , _ = g(z_int , y , reverse = False)
    loss = MSE(z , z_hat)
    end = time()
    
    print(-start + end)
    print(loss)
    
    
if __name__ == '__main__':
    unit_test_generator()
    # unit_test_latent_generator()