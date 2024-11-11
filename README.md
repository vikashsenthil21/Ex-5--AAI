<H3>NAME: VIKASH S</H3>
<H3>REGISTER NO. 212222240115</H3>
<H3>EX. NO.4</H3>
<H3>DATE: 11.09.2024</H3>
<H1 ALIGN =CENTER> Implementation of Hidden Markov Model</H1>

## Aim: 
Construct a Python code to find the sequence of hidden states by the known sequence of observances using Hidden Markov Model. Consider two hidden states Sunny and Rainy with observable states,happy and sad.

## Algorithm:

Step 1:Define the transition matrix, which specifies the probability of transitioning from  one hidden state to another.<br>
Step 2:Define the emission matrix, which specifies the probability of observing each possible observation given each hidden state.<br>
Step 3:Define the initial probabilities, which specify the probability of starting in each possible hidden state.<br>
Step 4:Define the observed sequence, which is the sequence of observations need to  be analyzed.<br>
Step 5:Initialize the alpha matrix with zeros, where each row represents a time step and each column represents a possible hidden state.<br>
Step 6:Calculate the first row of the alpha matrix by multiplying the initial  probabilities by the emission probabilities for the first observation.<br>
Step 7:Loop through the rest of the observed sequence and calculate the rest of the alpha matrix by multiplying the emission probabilities by the sum of the product of 
       the previous row of the alpha matrix and the corresponding row of the transition matrix.<br>
Step 8:Calculate the probability of the observed sequence by summing the last row of the alpha matrix.<br>
Step 9:Find the most likely sequence of hidden states by selecting the hidden state with the highest probability at each time step based on the alpha matrix.<br>

## Program:
```
import numpy as np
class KalmanFilter:
  def __init__ (self,F,H,Q,R,x0,P0):
    self.F = F #state transition model
    self.H = H # observation model
    self.Q = Q # process noise covariance
    self.R = R # measurement noise covariance
    self.x = x0 # initial state extimate
    self.P = P0 # initial error covariance
  
  def predict(self):
    #predict the next state
    self.x = np.dot(self.F, self.x)
    self.P = np.dot(np.dot(self.F, self.P),self.F.T) + self.Q
  
  def update(self, z):
    #update the state estimate based on the measurement z
    y = z - np.dot(self.H, self.x)
    S = np.dot(np.dot(self.H, self.P),self.H.T) + self.R
    K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
    self.x = self.x + np.dot(K, y)

#Example usage:
#Assume we want to track the position and velocity of a moving object
#with a state vector of [position, velocity] and a single scalar measurement
#of position.
dt = 0.1 # time step
F = np.array([[1, dt], [0, 1]]) # state transition model
H = np.array([[1, 0]]) # observation model
Q = np.diag([0.1, 0.1]) # process noise covariance
R = np.array([[1]]) # measurement noise covariance
x0 = np.array([0, 0]) # initial state estimate
P0 = np.diag([1, 1]) # initial error covariance


kf = KalmanFilter(F,H,Q,R,x0,P0)

true_states=[]
measurements=[]
for i in range(100):
  true_states.append([i*dt, 1]) #assume constant velocity of 1m/s
  measurements.append(i*dt + np.random.normal(scale=1)) # add measurement noise

#run the Kalman filter on the simulated measurements
est_states = []
for z in measurements:
    kf.predict()
    kf.update(np.array([z]))
    est_states.append(kf.x)

    

#plot the true and estimated positions
import matplotlib.pyplot as plt
plt.plot([s[0] for s in true_states], label='true')
plt.plot([s[0] for s in est_states], label='estimate')
plt.legend()
plt.show()
## Output:
## probability of the observed sequence
![image](https://github.com/user-attachments/assets/047fb3a9-83e2-4c2b-a0a4-2e6231b56593)

## Print the most likely sequence of weather states
![image](https://github.com/user-attachments/assets/0ec9dbd6-6c7d-4e83-8db9-d7deb6b26e19)
```

## Result:
Thus Hidden Markov Model is implemented using python.
