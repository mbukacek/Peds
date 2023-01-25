import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from matplotlib import pyplot as plt

#-----------------------------------#
#           DATA PREPARATION        # 
#-----------------------------------#

positions = pd.read_csv(r'source_data\bottleneck_43ped_4.csv')
positions.rename(columns = {'time(s)': 'time', 'x(cm)': 'x', 'y(cm)': 'y', 'density(m-2)': 'density'}, inplace = True)

detector = [0,150,-75,75]       # x_1, x_2, y_1, y_2
attractor = [0,0]               # x,y
vedge_angle = np.pi/3           # 60 degree left and right



#-----------------------------------#
#     CALCULATE VORONOI DENSITY     # 
#-----------------------------------#

# TO DO


#-----------------------------------#
#    CALCULATE MINIMUM DISTANCE     # 
#-----------------------------------#
 
positions['angle'] = np.arctan((positions.x - attractor[0])/(positions.y - attractor[1]))                                       # angle toward the centre of exit
positions['a_dist'] = np.sqrt(pow((positions.x - attractor[0]), 2)  + pow((positions.y - attractor[1]), 2))                     # distance toward the centre of exit
                                     
positions_merged = pd.merge(positions, positions[['x', 'y', 'ped', 'time', 'angle', 'a_dist']], how = 'inner', on = 'time')     # merge all positions on the same frame
positions_merged = positions_merged.drop(positions_merged[positions_merged.ped_x == positions_merged.ped_y].index)              # remove identical records

positions_merged = positions_merged[(abs(positions_merged.angle_x - positions_merged.angle_y) < vedge_angle)                    # filter out pairs out of vedge 
                                      & (positions_merged.a_dist_x < positions_merged.a_dist_y)]                                # and pairs where first pedestrian is not closer

# -> if there is no other pedestrian within certain angle, given pedestrian is removed from the set   

positions_merged['dist']= np.sqrt(pow((positions_merged.x_x - positions_merged.x_y), 2)                                         # calculate distance to all on frame (in vedge)
                                  + pow((positions_merged.y_x - positions_merged.y_y), 2))       
positions_merged['dist_rank'] = positions_merged.groupby(['ped_x','time'])['dist'].rank(method="first", ascending=True)         # rank distances
positions_nearest = positions_merged[(positions_merged.dist_rank == 1)]                                                         # keep nearest 
positions_nearest = positions_nearest[['time', 'ped_x', 'type',  'x_x', 'y_x', 'density', 'dist']]                              # remove redundant columns
positions_nearest.rename(columns = {'ped_x': 'ped', 'x_x': 'x', 'y_x': 'y', 'dist': 'mindist'}, inplace = True)                 # rename columns

positions_nearest = positions_nearest[positions_nearest.density > 0]                                                            # filter out invalid densities

positions_nearest['mindist_inv'] = 1/positions_nearest.mindist
positions_nearest['mindist_inv2'] = 1/pow(positions_nearest.mindist,2)


#-----------------------------------#
#        IMPLEMENT DETECTOR         # 
#-----------------------------------#

# plot  
# plt.scatter(positions_dist.x, positions_dist.y, c = positions_dist.mindist, cmap='viridis', s = positions_dist.mindist)

positions_nearest_det = positions_nearest[(positions_nearest.x > detector[0]) & (positions_nearest.x < detector[1]) 
                               & (positions_nearest.y > detector[2]) & (positions_nearest.x < detector[3])]

positions_det = positions[(positions.x > detector[0]) & (positions.x < detector[1]) 
                          & (positions.y > detector[2]) & (positions.x < detector[3])]

tmp = positions_nearest_det.groupby(['time'], as_index=False).mean()                        # based on filtered sample
tmp2 = positions_det.groupby(['time'], as_index=False).count()                              # based on full sample

records_det = pd.merge(tmp[['time', 'density', 'mindist', 'mindist_inv', 'mindist_inv2']],
                        tmp2[['time', 'ped']], how = 'inner', on = 'time')


#-----------------------------------#
#           RUN REGRESSION          # 
#-----------------------------------#


plt.scatter(records_det.density, records_det.mindist_inv)

result = sm.ols(formula = "density ~ mindist_inv", data = records_det).fit()
#print(result.params)
print(result.summary())

# RAW DATA bottleneck_25ped_2
# mindist_inv:  Adj. R-squared:                  0.526
# mindist_inv2: Adj. R-squared:                  0.562

# DETECTOR DATA bottleneck_25ped_2
# mindist_inv:  Adj. R-squared:                  0.690
# mindist_inv2: Adj. R-squared:                  0.687

# DETECTOR DATA bottleneck_43ped_4
# mindist_inv:  Adj. R-squared:                  0.837
# mindist_inv2: Adj. R-squared:                  0.800

# DETECTOR DATA FILTERED bottleneck_43ped_4
# mindist_inv:  Adj. R-squared:                  0.819
# mindist_inv2: Adj. R-squared:                  0.813











