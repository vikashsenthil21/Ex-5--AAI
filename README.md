
<H3>ENTER YOUR NAME: VIKASH S</H3>
<H3>ENTER YOUR REGISTER NO.212222240115</H3>
<H3>EX. NO.5</H3>
<H3>DATE: 14.09.2024</H3>
<H1 ALIGN =CENTER> Implementation of Kalman Filter</H1>
<H3>Aim:</H3> To Construct a Python Code to implement the Kalman filter to predict the position and velocity of an object.
<H3>Algorithm:</H3>
Step 1: Define the state transition model F, the observation model H, the process noise covariance Q, the measurement noise covariance R, the initial state estimate x0, and the initial error covariance P0.<BR>
Step 2:  Create a KalmanFilter object with these parameters.<BR>
Step 3: Simulate the movement of the object for a number of time steps, generating true states and measurements. <BR>
Step 3: For each measurement, predict the next state using kf.predict().<BR>
Step 4: Update the state estimate based on the measurement using kf.update().<BR>
Step 5: Store the estimated state in a list.<BR>
Step 6: Plot the true and estimated positions.<BR>

<H3>Program:</H3>

```python

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

```

<H3>Output:</H3>

![image](https://github.com/user-attachments/assets/656448a0-8758-49b8-8647-d4d880f9a1cb)



<H3>Results:</H3>
Thus, Kalman filter is implemented to predict the next position and  velocity in Python
