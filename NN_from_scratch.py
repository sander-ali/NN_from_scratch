import numpy as np
import matplotlib.pyplot as plt
def func_sig(x):
  return 1 / (1 + np.exp(-x))

def func_sig_der(x):
  f_x = func_sig(x)
  return f_x * (1 - f_x)

def loss_mse(yt, yp):
  return ((yt - yp) ** 2).mean()

class NN_from_scratch:
  '''
  This is a neural network having an input layer, a hidden layer with 2 units
  and an output layer.
 
  Note: This function is only written to make the students understand how neural
  networks work. The implementation in packages is quite difference and is optimal
  in comparison to this code so do not use this code if you need optimal or 
  more accurate results. 
  '''
  def __init__(self):
    # Weights
    self.w_1 = np.random.normal()
    self.w_2 = np.random.normal()
    self.w_3 = np.random.normal()
    self.w_4 = np.random.normal()
    self.w_5 = np.random.normal()
    self.w_6 = np.random.normal()

    # Biases
    self.b_1 = np.random.normal()
    self.b_2 = np.random.normal()
    self.b_3 = np.random.normal()

  def FFNet(self, x):
    h_1 = func_sig(self.w_1 * x[0] + self.w_2 * x[1] + self.b_1)
    h_2 = func_sig(self.w_3 * x[0] + self.w_4 * x[1] + self.b_2)
    o_1 = func_sig(self.w_5 * h_1 + self.w_6 * h_2 + self.b_3)
    return o_1

  def train_Network(self, data, Labels):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - Labels is a numpy array with n elements.
      Elements in Labels correspond to those in data.
    '''
    LR = 0.1
    epochs = 500 # number of times to loop through the entire dataset
    cum_loss = []

    for epoch in range(epochs):
      for x, yt in zip(data, Labels):
        sum_h1 = self.w_1 * x[0] + self.w_2 * x[1] + self.b_1
        h_1 = func_sig(sum_h1)

        sum_h2 = self.w_3 * x[0] + self.w_4 * x[1] + self.b_2
        h_2 = func_sig(sum_h2)

        sum_o1 = self.w_5 * h_1 + self.w_6 * h_2 + self.b_3
        o_1 = func_sig(sum_o1)
        yp = o_1

        #computing partial derivatives
        # convention: pd_L_pd_w_1 represents "partial L / partial w_1"
        pd_L_pd_yp = -2 * (yt - yp)
        
        pd_yp_pd_w_5 = h_1 * func_sig_der(sum_o1)
        pd_yp_pd_w_6 = h_2 * func_sig_der(sum_o1)
        pd_yp_pd_b_3 = func_sig_der(sum_o1)

        pd_yp_pd_h_1 = self.w_5 * func_sig_der(sum_o1)
        pd_yp_pd_h_2 = self.w_6 * func_sig_der(sum_o1)

        pd_h_1_pd_w_1 = x[0] * func_sig_der(sum_h1)
        pd_h_1_pd_w_2 = x[1] * func_sig_der(sum_h1)
        pd_h_1_pd_b_1 = func_sig_der(sum_h1)

        pd_h_2_pd_w_3 = x[0] * func_sig_der(sum_h2)
        pd_h_2_pd_w_4 = x[1] * func_sig_der(sum_h2)
        pd_h_2_pd_b_2 = func_sig_der(sum_h2)

        # --- Update weights and biases
        self.w_1 -= LR * pd_L_pd_yp * pd_yp_pd_h_1 * pd_h_1_pd_w_1
        self.w_2 -= LR * pd_L_pd_yp * pd_yp_pd_h_1 * pd_h_1_pd_w_2
        self.b_1 -= LR * pd_L_pd_yp * pd_yp_pd_h_1 * pd_h_1_pd_b_1

        self.w_3 -= LR * pd_L_pd_yp * pd_yp_pd_h_2 * pd_h_2_pd_w_3
        self.w_4 -= LR * pd_L_pd_yp * pd_yp_pd_h_2 * pd_h_2_pd_w_4
        self.b_2 -= LR * pd_L_pd_yp * pd_yp_pd_h_2 * pd_h_2_pd_b_2

        self.w_5 -= LR * pd_L_pd_yp * pd_yp_pd_w_5
        self.w_6 -= LR * pd_L_pd_yp * pd_yp_pd_w_6
        self.b_3 -= LR * pd_L_pd_yp * pd_yp_pd_b_3
        
      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        yp = np.apply_along_axis(self.FFNet, 1, data)
        loss = loss_mse(Labels, yp)
        cum_loss.append(loss)
        print("Epoch %d loss: %.3f" % (epoch, loss))
    plt.plot(cum_loss, 'g--', linewidth=2)
    plt.title('Plotting Loss Function')
    plt.show()

# Define dataset
data = np.array([
  [-2, -1],  # Shuri
  [25, 6],   # Bucky
  [17, 4],   # Charlie
  [-15, -6], # Mira
  [-9, -2], #Natalia
  [20, 5], #Roger
  [-4, -1], #Natasha
  [-8, -6], #Gamora
  [-12, -3], #Nebula
  [-13, -2], #Brie
  [-10, -5], #Mitchell
  [15, 2], #Tony
  [30, 8], #Steve
  [50, 15], #Bruce
  [10, 3], #Paul
  [5, 3], #Barton
])
Labels = np.array([
  1, # Shuri
  0, # Bucky
  0, # Charlie
  1, # Mira
  1, #Natalia
  0, #Roger
  1, #Natasha
  1, #Gamora
  1, #Nebula
  1, #Brie
  1, #Mitchell
  0, #TOny
  0, #Steve
  0, #Bruce
  0, #Paul
  0, #Barton
])

# Train our neural network!
network = NN_from_scratch()
network.train_Network(data, Labels)

# Predicting the gender
Okoye = np.array([-7, -3]) # 128 pounds, 63 inches
Stephen = np.array([20, 2])  # 155 pounds, 68 inches
print("Okoye: %.3f" % network.FFNet(Okoye)) # 0.975 - F
print("Frank: %.3f" % network.FFNet(Stephen)) # 0.026 - M