# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:33:49 2023

@author: Rik
"""

import wis_2_2_utilities as util
import wis_2_2_systems as systems
import matplotlib.pyplot as plt
import pandas as pd
import control as ct
import numpy as np

# set constants
timestep = 2e-3
rho_wood = 700
g = 9.81
breedte = 0.02  # m
num_pendulum = 2
pendulum_length = 0.6
hoogte = 0.02
m = pendulum_length * breedte * rho_wood * hoogte

I = 1/12 * rho_wood * breedte**4 * pendulum_length + 1/3 * rho_wood * breedte**2 * pendulum_length**3
Icm =  1/12 * rho_wood * breedte**2* pendulum_length * (breedte**2 + pendulum_length**2)

A = np.array([[1, 0, 0, 0],
              [0, I+1/2*m*pendulum_length**2+m*pendulum_length**2/4, 0,m*pendulum_length**2/4],
              [0, 0, 1, 0],
              [0, Icm+1/8*m*pendulum_length**2+m*pendulum_length**2/4 , 0, Icm+1/8*m*pendulum_length**2]])
B = np.array([[0,1, 0,0],
              [m*g*pendulum_length, 0, 0, 0],
              [0, 0,0, 1],
              [ m*g*pendulum_length/2, 0,  m*g*pendulum_length/2, 0]])
C = np.array([[0],
              [1],
              [0],
              [0]])
# 
# Define matrices A and B based on the given equations
matrix_A = np.dot(np.linalg.inv(A), B)
matrix_B = np.dot(np.linalg.inv(A), C)
# print(matrix_A)
# print(matrix_B)

# Controller inleveropdracht 2
class controller():
    def __init__(self, target=0):

        # Desired pole locations
        desired_poles = [-19.7, -20.7, -21.7, -22.7]

        matrix_K = ct.place(matrix_A, matrix_B, desired_poles)
        self.matrix_k = matrix_K
        eigenvaluesK= np.linalg.eigvals(matrix_A)
        #print("Eigenwaarden:", eigenvaluesK)
        print("matrix K:", matrix_K)

    def feedBack(self, observe):
        u = -self.matrix_k @ observe
        return u

def datawork(): 
    """
    Created on Thur 16-11-2023 15:24

    @author: Nathan
    """
    # Define matrices A and B based on the given equations
    matrix_A = np.dot(np.linalg.inv(A), B)
    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix_A)
    #print("Eigenwaarden:", eigenvalues)
        
    matrix_B = np.dot(np.linalg.inv(A), C)

    # Controllability
    wc = ct.ctrb(matrix_A, matrix_B)
    controllability_rank = np.linalg.matrix_rank(wc)
    print("Controleerbaarheid:", "Volledig controleerbaar" if controllability_rank == matrix_A.shape[0] else "Niet volledig controleerbaar")

    # Stabilizability
    uncontrollable_modes = eigenvalues[controllability_rank:]
    print("Stabiliseerbaarheid:", "Stabiliseerbaar" if all(np.real(uncontrollable_modes) < 0) else "Niet stabiliseerbaar")

    # Observability
    matrix_C = np.eye(4)
    wo =ct.obsv(matrix_A, matrix_C)
    observability_rank = np.linalg.matrix_rank(wo)
    print("Observeerbaarheid:", "Volledig observeerbaar" if observability_rank == matrix_A.shape[0] else "Niet volledig observeerbaar")

    # Detectability
    unobservable_modes = eigenvalues[observability_rank:]
    print("Detecteerbaarheid:", "Detecteerbaar" if all(np.real(unobservable_modes) > 0) else "Niet detecteerbaar")

    # Read the data
    data = pd.read_csv('stacked_inverted_pendulum.csv', sep=',')
    
    # Define your column names
    column_names = ['tijd', 'kwad_toestand_kosten', 'kwad_input_kosten', 'hoek_slinger1','hoeksnelheid_slinger1','hoek_slinger_2','hoeksnelheid_slinger_2','input'] 
    
    # Assign the column names to the DataFrame
    data.columns = column_names
    
    tijd = data['tijd']
    toestand_kosten = data['kwad_toestand_kosten']
    input_kosten = data['kwad_input_kosten']
    hoek_1 = data['hoek_slinger1']
    hoeksnelheid_1 = data['hoeksnelheid_slinger1']
    hoek_2 = data['hoek_slinger_2']
    hoeksnelheid_2 = data['hoeksnelheid_slinger_2']
    inputs = data['input']
    
    # Find point where system became stable 
    snijpunten_1 = []
    snijpunten_2 = []
    snijpunten_1.append([0.0, 0.0])
    snijpunten_2.append([0.0, 0.0])
    for i in range(1, len(tijd)):
        if (-0.1 >=hoeksnelheid_1[i] or 0.1 <= hoeksnelheid_1[i]):
            snijpunten_1.append([tijd[i], hoeksnelheid_1[i]])
        if (-0.1 >=hoeksnelheid_2[i] or 0.1 <= hoeksnelheid_2[i]):
            snijpunten_2.append([tijd[i], hoeksnelheid_2[i]])

    print("Sijpunten Pendulum 1:",snijpunten_1[-1])
    print("Sijpunten Pendulum 2:",snijpunten_2[-1])
    
    #plot
    plt.plot(tijd, inputs)
    plt.xlabel('Tijd(s)')
    plt.ylabel('Input')
    plt.title('Input over tijd')
    plt.show()
    
        
    #plot
    plt.plot(tijd, hoeksnelheid_1)
    plt.xlabel('Tijd(s)')
    plt.ylabel('Snelheid slinger 1')
    plt.title('Snelheid over tijd van Pendulum 1')
    plt.scatter(snijpunten_1[-1][0], snijpunten_1[-1][1], color= 'g')
    # Add red lines at y = 0.1 and y = -0.1
    plt.axhline(y=0.1, color='red', linestyle='--', label='0.1')
    plt.axhline(y=-0.1, color='red', linestyle='--', label='-0.1')
    plt.show()
    
    #plot
    plt.plot(tijd, hoeksnelheid_2)
    plt.xlabel('Tijd(s)')
    plt.ylabel('Snelheid slinger 2')
    plt.title('Snelheid over tijd van Pendulum 2')
    plt.scatter(snijpunten_2[-1][0], snijpunten_2[-1][1], color= 'g')     
    # Add red lines at y = 0.1 and y = -0.1
    plt.axhline(y=0.1, color='red', linestyle='--', label='0.1')
    plt.axhline(y=-0.1, color='red', linestyle='--', label='-0.1')
    
    
    # Set upper and lower limits for the y-axis
    plt.ylim(bottom=-0.5, top=0.5)
    plt.show()
    
def main():
  model=systems.stacked_inverted_pendulum(num_pendulum = 2)
  control = controller()
  simulation = util.simulation(model=model,timestep=timestep)
  simulation.setCost()
  simulation.max_duration = 10 #seconde
  simulation.GIF_toggle = False #set to false to avoid frame and GIF creation


  while simulation.vis.Run():
      if simulation.time<simulation.max_duration:
        simulation.step()
        u = control.feedBack(simulation.observe())
        simulation.control(u)
        simulation.log()
      else:
        print('Ending visualisation...')
        simulation.vis.GetDevice().closeDevice()
        	
  print("Kwadratische kosten:", simulation.cost_input)
        
  simulation.writeData()
  datawork()     
  
if __name__ == "__main__":
 main()
 