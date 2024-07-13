# Description: Main file to run the simulation
from imports import *
from functions import compare,VPG,SEVAC
from draw_from_file import draw
from generate_net import generate_all

def main():

    #file_path = 'Networks\Winnipeg\Winnipeg_network.csv'
    network='Anaheim'
    #od_file_path=''
    iterations = 300
    alpha = 0.01
    #target_path='Networks\Winnipeg\Winnipeg_network.csv'
    #VPG(file_path, iterations, alpha,19)
    #off_policy_VPG(file_path, iterations, alpha)
    #VAC(file_path, iterations, alpha)
    compare(network, iterations, alpha)
    #draw('output_data/arrays_data_T=10.json')
    #generate_all(file_path,target_path,0.1)
    #SEVAC(file_path, iterations, alpha,time_budget)

if __name__ == "__main__":
    main()
