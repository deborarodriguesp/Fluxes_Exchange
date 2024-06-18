import sys
import h5py
import os
import numpy as np
import pandas as pd
import datetime
import math

############### Load HDF files ###############

# Specify the input folder containing HDF files
input_folder = r'D:\DOUTORAMENTO\MOHID_Foz\Results\HDF'
output_folder = r'E:\Debora\Fluxes_soil_river'

# My original line are the elements in contact with the Hydrodynamic grid.
line_original_file = r'D:\DOUTORAMENTO\MOHID_Foz\Fluxes_soil_river\Hidrodinamic_shape\Teste_menos_simplificado\teste_menos_simp_line_cut_selected_elements_centroid_coordinates.csv'

# My parallel line is the selection of elements using v.parallel from QGIS.
# Selecting the second elements line in contact with the hydrodynamic grid.
line_parallel_file = r'D:\DOUTORAMENTO\MOHID_Foz\Fluxes_soil_river\Hidrodinamic_shape\Teste_menos_simplificado\teste_menos_simp_line_cut_parallel_selected_elements_centroid_coordinates.csv'

ij_indexes_relation = True

water_column_data = {}

# funcao(argumentos)
def original_read(line_original_file):
    fin = open(line_original_file, 'r')
    line_original = fin.readlines()[1:]
    fin.close()

    return line_original

def parallel_read(line_parallel_file):
    fin = open(line_parallel_file, 'r')
    line_parallel_data = fin.readlines()[1:]
    fin.close()
    #    return line_parallel
    line_parallel = []
    for line in line_parallel_data:
        # Remove leading and trailing brackets and whitespace
        line = line.strip().strip('[]')
        values = line.split(',')

        line_parallel.append(values)

    return line_parallel

def select_data(line_original, line_parallel):
    selected_data = []

    for original_line in line_original:
        i_ori, j_ori, dtm_ori, x_ori, y_ori = original_line.split(';')

        distance_list = []
        
        for parallel_line in line_parallel:
            parallel_values = parallel_line[0].split(';')
            i_par, j_par, dtm_par, x_par, y_par = parallel_values

            # Euclidean distance between the coordinates
            distance = math.sqrt((float(x_ori) - float(x_par)) ** 2 + (float(y_ori) - float(y_par)) ** 2)
            distance_round = round(distance,5)
            distance_list.append([distance_round,i_par,j_par])

        first_column = [sublist[0] for sublist in distance_list]
        min_value=min(first_column)
        sel_indexes=[[x[1],x[2]] for x in distance_list if x[0] == min_value]
        
        for element in sel_indexes:
            i_closest = element[0]
            j_closest = element[1]
            selected_data.append((i_ori+','+j_ori + ';' + i_closest +',' + j_closest))
    return selected_data

def extract_time(hdf_fin, option):

    formatted_times = []
    time_group = hdf_fin['Time']
    
    for d, dataset_name in enumerate(time_group):
        dataset = time_group[dataset_name]
        time_data = dataset[:]

        if option=='first_file' and d==0:
            global initial_instant
            initial_instant = datetime.datetime(int(time_data[0]), int(time_data[1]), int(time_data[2]), int(time_data[3]), int(time_data[4]), int(time_data[5]))
            
        current_instant = datetime.datetime(int(time_data[0]), int(time_data[1]), int(time_data[2]), int(time_data[3]), int(time_data[4]), int(time_data[5]))
        delta_time = (current_instant - initial_instant).total_seconds()
        formatted_times.append(delta_time)

    return formatted_times

def extract_properties(hdf_fin, prop):
    prop_values = []
    # See what the HDF file has as groups:
    result_hdf = hdf_fin.get('Results')
    prop_group = result_hdf.get(prop)
    
    # Iterate over each dataset in the flow_group
    for dataset_name, dataset in prop_group.items():
        # Access the flow data for the current dataset
        prop_data = dataset[:]
        prop_data = prop_data.round(2)
        #Rotation 90 degree the matrix
        prop_data = np.rot90(prop_data)
        prop_values.append(prop_data)

    return prop_values

################## MAIN ##################

if ij_indexes_relation:

    # Read the original and parallel lines
    line_original = original_read(line_original_file)
    line_parallel = parallel_read(line_parallel_file)

    # Select the data based on the original and parallel lines
    selected_data = select_data(line_original, line_parallel)
    output_file = os.path.join(output_folder, "selected_data.csv")
    np.savetxt(output_file, selected_data, delimiter=',', fmt='%s')

# Read index (ij) file
fin = open('selected_data.csv', 'r')
ij_from_file = fin.readlines()
fin.close()

original_line_i = []
original_line_j = []
parallel_line_i = []
parallel_line_j = []

for lin in ij_from_file:
    o, p = lin.split(';')
    io, jo = o.split(',')
    ip, jp = p.split(',')
    
    original_line_i.append(int(io))
    original_line_j.append(int(jo))  
    parallel_line_i.append(int(ip))
    parallel_line_j.append(int(jp)) 

# Fill in the list of HDF files here
folders=os.listdir(input_folder)
folders.sort()

# Lists
time_list = []
water_column_list = []
flow_modulus_list = []

for direc in folders[20:22]:
    hdf_file = input_folder+ '/' +direc+ '/' + 'RunOff_1.hdf5'
    print ('working on file ' + direc)
    
    with h5py.File(hdf_file, 'r') as hdf_fin:
    # Extract the time from the HDF files
        if time_list == []:
            formatted_times = extract_time(hdf_fin, 'first_file')
            time_list = time_list+formatted_times[:-1]
        else:
            formatted_times = extract_time(hdf_fin, '')
            #print(type(formatted_times))
            time_list = time_list+formatted_times[:-1]
        
        #extract water level and flow modulus
        water_column = extract_properties(hdf_fin, 'water level')
        water_column_list = water_column_list+water_column[:-1]
        
        flow_modulus = extract_properties(hdf_fin, 'flow modulus')
        flow_modulus_list = flow_modulus_list+flow_modulus[:-1]
                
        hdf_fin.close()
        
for ij_indexes in zip(original_line_i, original_line_j,parallel_line_i,parallel_line_j):
    for inst, wl_instant in enumerate(water_column_list):
        discharge = 0
        wl_original = wl_instant[wl_instant.shape[0]-ij_indexes[0],ij_indexes[1]-1]
        wl_parallel = wl_instant[wl_instant.shape[0]-ij_indexes[2],ij_indexes[3]-1]
        
        if wl_parallel > wl_original:
            discharge = flow_modulus_list[inst][wl_instant.shape[0]-ij_indexes[2],ij_indexes[3]-1]  

# Create an empty DataFrame with 'Time' as the first column
discharge_df = pd.DataFrame({'Time': time_list})

# Create column headers using the IJ pairs from parallel lines
column_headers = [f'{i},{j}' for i, j in zip(parallel_line_i, parallel_line_j)]

# Initialize an empty dictionary to store the data
data = {'Time': time_list}

# Loop through the original and parallel lines to associate discharge values with the respective IJ pairs
for ij_indexes in zip(original_line_i, original_line_j, parallel_line_i, parallel_line_j):
    #Create a Column name
    column_name = f'{ij_indexes[2]},{ij_indexes[3]}'  # print this gives me 903 IJ pairs
    discharge_values = []  # List to store the discharge values for each time step
    
    for inst, wl_instant in enumerate(water_column_list):
        discharge = 0
        wl_original = wl_instant[wl_instant.shape[0] - ij_indexes[0], ij_indexes[1] - 1]
        wl_parallel = wl_instant[wl_instant.shape[0] - ij_indexes[2], ij_indexes[3] - 1]

        if wl_parallel > wl_original:
            discharge = flow_modulus_list[inst][wl_instant.shape[0] - ij_indexes[2], ij_indexes[3] - 1]
        discharge_values.append(discharge)

    data[column_name] = discharge_values

# Convert the dictionary to a DataFrame
discharge_df = pd.DataFrame(data)

initial_instant_str = initial_instant.strftime('%Y %m %d %H %M %S')

# Construct MOHID time series
for ij_indexes in zip(original_line_i, original_line_j, parallel_line_i, parallel_line_j):
    file_to_create = f'flow_modulus_{ij_indexes[2]}_{ij_indexes[3]}.ets'
    with open(file_to_create, 'w') as mohid_file:
        mohid_file.write("NAME                    : " + file_to_create.replace('.ets', '') + '\n')
        mohid_file.write("SERIE_INITIAL_DATA      : " + initial_instant_str + '\n')
        mohid_file.write("TIME_UNITS              : DAYS\n")
        mohid_file.write("!time flow_modulus\n")
        mohid_file.write("<BeginTimeSerie>\n")
        for index, row in discharge_df.iterrows():
            column_name = f'{ij_indexes[2]},{ij_indexes[3]}'
            mohid_file.write(str(row['Time']) + ' ' + str(row[column_name]) + '\n')
        mohid_file.write("<EndTimeSerie>")

# Open the discharge.dat file
with open("Discharge_1.dat", 'w') as mohid_dat_final:
    # Loop through the original and parallel lines to create the discharge blocks
    for ij_indexes in zip(original_line_i, original_line_j, parallel_line_i, parallel_line_j):
        mohid_dat_final.write("<begindischarge>\n")
        mohid_dat_final.write("NAME                    : " + f'{ij_indexes[2]}_{ij_indexes[3]}\n')
        mohid_dat_final.write("DESCRIPTION             : Descarga Climatológica\n")
        mohid_dat_final.write("DEFAULT_FLOW_VALUE      : 0\n")
        mohid_dat_final.write("I_CELL                  : " + f'{ij_indexes[2]}\n')
        mohid_dat_final.write("J_CELL                  : " + f'{ij_indexes[3]}\n')
        mohid_dat_final.write("K_CELL                  : 0\n")
        mohid_dat_final.write("DATA_BASE_FILE          : " + f'../GeneralData/Discharge/flow_modulus_{ij_indexes[2]}_{ij_indexes[3]}.ets\n')
        mohid_dat_final.write("USE_ORIGINAL_VALUES     : 1\n")
        mohid_dat_final.write("FLOW_COLUMN             : 2\n")
        #mohid_dat_final.write("SPATIAL_EMISSION        : Line\n")
        #mohid_dat_final.write("SPATIAL_FILE            : " + f'..\GeneralData\Discharge\Q{ij_indexes[2]}_{ij_indexes[3]}.lin\n')
        mohid_dat_final.write("<enddischarge>\n")
        mohid_dat_final.write("\n")
