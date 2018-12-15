import numpy as np
import pandas as pd
from collections import defaultdict

from math import sin, cos, sqrt, atan2, radians




data_dir = "./data/"

# Reading all the branch pincodes and their respective addresses
tot_branch_pincodes = pd.read_csv(data_dir+'Branch_pincode_address_fsfb.csv', dtype = str)


# Reading all the pincodes and their latitudes and longitudes for India from geonames
tot_india_pincodes = pd .read_csv(data_dir+'pincode_lat_long_geonames.csv', dtype = str)

# Validate if all the given branch bank pincodes are valid
# print(tot_branch_pincodes.loc[tot_branch_pincodes['adj_pincode'].isin(set(tot_india_pincodes['Postal Code'])) == False,"adj_pincode"])

# Ignoring the invalid pincode and storing the valid branch pin codes in 'tot_valid_branch_pincode'
tot_valid_branch_pincode = tot_branch_pincodes.loc[tot_branch_pincodes['adj_pincode'].isin(set(tot_india_pincodes['Postal Code'])) == True,:]

# Checking if there more than one branches in the same pincode. If so it has to be validated.
val, count = np.unique(tot_branch_pincodes['adj_pincode'], return_counts=True)
for idx,ind_count in enumerate(count):
    if ind_count > 1:
        val[idx]

# Reading in all the customers pincode.
tot_cust_pincode = pd.read_csv(data_dir+'Registered_Customers_pincode_address_fsfb.csv', dtype = str)

# Ignoring all the invalid pincodes and storing all the customers with valid pin codes on 'tot_valid_cust_pincode' dataframe
tot_valid_cust_pincode = tot_cust_pincode.loc[tot_cust_pincode['capin'].isin(set(tot_india_pincodes['Postal Code'])) == True,:]

# TODO remove this line of code. Temp purpose only
#tot_valid_cust_pincode = tot_valid_cust_pincode.iloc[:10, :]

# map each customer and branch pincodes with latitude and longitude
# consolidating customer and branch pincodes into one series using append
all_bank_valid_pincodes = tot_valid_cust_pincode['capin'].append(tot_valid_branch_pincode['adj_pincode'],ignore_index = True)

# converting the series into a list of unique pincodes
all_unique_valid_pincodes = list(np.unique(all_bank_valid_pincodes))

# creating a new int default dict called 'pincode_map_lat_long' and mapping each
# pincode to its latitude and longitude
pincode_map_lat_long = defaultdict(int)
for pincode in all_unique_valid_pincodes:
    df = tot_india_pincodes.loc[tot_india_pincodes['Postal Code'] == pincode,["latitude","longitude"]].reset_index(drop = True)
    pincode_map_lat_long[int(pincode)] = [float(df.iloc[0,0]), float(df.iloc[0,1])]


def find_closest_branch(cust_pincode):
    '''

    :param cust_pincode: Customer pincode for which nearest branch needs to be found
    :return: return nearest branch pincode and distance to it from customer pincode
    '''

    # TODO since it is a puny dataset the code has not been optimised.
    distance_matrix = []
    lat_1, lon_1 = pincode_map_lat_long[int(cust_pincode)]
    temp_lst = list(tot_valid_branch_pincode['adj_pincode'])
    for branch_pincode in temp_lst:
        lat_2, lon_2 = pincode_map_lat_long[int(branch_pincode)]

        # approximate radius of earth in km
        R = 6373.0

        lat1 = radians(lat_1)
        lon1 = radians(lon_1)
        lat2 = radians(lat_2)
        lon2 = radians(lon_2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # Formula to find the distance between two latitudes and longitudes
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        if distance == 0:
            return branch_pincode, 0

        distance_matrix.append((cust_pincode, branch_pincode, distance))

    df = pd.DataFrame(distance_matrix, columns=['customer_pincode', 'branch_pincode', 'distance']).reset_index(drop=True)
    closest_destination = df.loc[df['distance'] == min(df['distance']), :]
    closest_destination = closest_destination.iloc[0,:]
    return (closest_destination[1], closest_destination[2])


# using apply method to go through customer pincode one and one and find the nearest branch
tot_valid_cust_pincode['pincode_distance'] =  tot_valid_cust_pincode['capin'].apply(lambda x: find_closest_branch(x))

# splitting the tuple in one of the columns('pincode_distance') into two different columns
tot_valid_cust_pincode[['branch_pincode', 'distance']] = tot_valid_cust_pincode['pincode_distance'].apply(pd.Series)

# droppping the original column 'pincode_distance' from the dataframe
tot_valid_cust_pincode.drop('pincode_distance', axis = 1, inplace = True)

# writing the final results in a .csv file
tot_valid_cust_pincode.to_csv(data_dir+'final.csv', index = False)


















