import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from matplotlib import pyplot as plt

#-----------------------------------#
#           DATA PREPARATION        # 
#-----------------------------------#

positions = pd.read_csv(r'source_data\bottleneck_25ped_2.csv')
positions.rename(columns = {'time(s)': 'time', 'x(cm)': 'x', 'y(cm)': 'y', 'density(m-2)': 'density'}, inplace = True)


#-----------------------------------#
#    CALCULATE MINIMUM DISTANCE     # 
#-----------------------------------#
 
positions_merged = pd.merge(positions, positions[['x', 'y', 'ped', 'time']], how = 'inner', on = 'time')                        # merge all positions on the same frame
positions_merged = positions_merged.drop(positions_merged[positions_merged.ped_x == positions_merged.ped_y].index)              # remove identical records

# TO DO: incororate only pedestrians in certain angle 
#            towards the exit  
#            towards the head orientation  
#            towards the velocity vector in front
#         .. first proxy : closed to the exit ?

# TO DO: if the nearest in certain angle is not found, dist = inf .. handle later in regression?   

positions_merged['dist']= np.sqrt(pow((positions_merged.x_x - positions_merged.x_y), 2)                                         # calculate distance to all on frame 
                                  + pow((positions_merged.y_x - positions_merged.y_y), 2))       
positions_merged['dist_rank'] = positions_merged.groupby(['ped_x','time'])['dist'].rank(method="first", ascending=True)         # rank distances
positions_dist = positions_merged[(positions_merged.dist_rank == 1)]                                                            # keep nearest 
positions_dist = positions_dist[['time', 'ped_x', 'type',  'x_x', 'y_x', 'density', 'dist']]                                    # remove redundant columns
positions_dist.rename(columns = {'ped_x': 'ped', 'x_x': 'x', 'y_x': 'y', 'dist': 'mindist'}, inplace = True)                    # rename columns

positions_dist = positions_dist[positions_dist.density > 0]

positions_dist['mindist_inv'] = 1/positions_dist.mindist
positions_dist['mindist_inv2'] = 1/pow(positions_dist.mindist,2)


#-----------------------------------#
#        IMPLEMENT DETECTOR         # 
#-----------------------------------#

# plot  
# plt.scatter(positions_dist.x, positions_dist.y, c = positions_dist.mindist, cmap='viridis', s = positions_dist.mindist)

position_det = positions_dist[(positions_dist.x > 0) & (positions_dist.x < 150) 
                               & (positions_dist.y > -75) & (positions_dist.x < 75)]

tmp = position_det.groupby(['time'], as_index=False).mean()
tmp2 = position_det.groupby(['time'], as_index=False).count()

position_det = pd.merge(tmp[['time', 'density', 'mindist', 'mindist_inv', 'mindist_inv2']],
                        tmp2[['time', 'ped']], how = 'inner', on = 'time')


#-----------------------------------#
#           RUN REGRESSION          # 
#-----------------------------------#


plt.scatter(position_det.density, position_det.mindist_inv)

result = sm.ols(formula = "density ~ mindist_inv", data = position_det).fit()
#print(result.params)
print(result.summary())

# RAW DATA bottleneck_25ped_2
# mindist_inv:  Adj. R-squared:                  0.526
# mindist_inv2: Adj. R-squared:                  0.562

# DETECTOR DATA bottleneck_25ped_2
# mindist_inv:  Adj. R-squared:                  0.690
# mindist_inv2: Adj. R-squared:                  0.687















