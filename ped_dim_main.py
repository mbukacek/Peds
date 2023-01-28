import pandas as pd
import numpy as np
import statsmodels.formula.api as sm          # https://www.statsmodels.org/
from matplotlib import pyplot as plt 



def calculate_min_dist_vedge(positions):
    
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
    positions_nearest['mindist_log'] = np.log(positions_nearest.mindist)
    
    return positions_nearest


def calculate_detector_means(positions_nearest, positions, detector):
    
    positions_nearest_det = positions_nearest[(positions_nearest.x > detector[0]) & (positions_nearest.x < detector[1]) 
                                   & (positions_nearest.y > detector[2]) & (positions_nearest.x < detector[3])]

    positions_det = positions[(positions.x > detector[0]) & (positions.x < detector[1]) 
                              & (positions.y > detector[2]) & (positions.x < detector[3])]

    tmp = positions_nearest_det.groupby(['time'], as_index=False).mean()                        # based on filtered sample (only peds with nereast in vedge)
    tmp.rename(columns = {'density': 'density_mean',
                          'mindist': 'mindist_mean', 
                          'mindist_inv': 'mindist_inv_mean', 
                          'mindist_inv2': 'mindist_inv2_mean',
                          'mindist_log': 'mindist_log_mean'
                          }, 
               inplace = True)

    tmp2 = positions_det.groupby(['time'], as_index=False).count()                              # based on full sample
    tmp2.rename(columns = {'ped': 'ped_cnt'}, inplace = True)

    records_det = pd.merge(tmp[['time', 'density_mean', 'mindist_mean', 'mindist_inv_mean', 'mindist_inv2_mean', 'mindist_log_mean']], 
                           tmp2[['time', 'ped_cnt']], how = 'inner', on = 'time')

    records_det['mindist_mean_inv'] = 1/records_det.mindist_mean
    records_det['mindist_mean_inv2'] = 1/pow(records_det.mindist_mean,2)
    records_det['mindist_mean_log'] = np.log(records_det.mindist_mean)
    records_det['density_mean_log'] = np.log(records_det.density_mean)
    
    return records_det


def run_regression(records_det, model, intercept):
    
    if intercept:
    
        result = sm.ols(formula = model, data = records_det).fit()

        regress_summary = pd.DataFrame({'experiment': experimental_code, 
                                    'exp_round': experimental_rounds[rn], 
                                    'model': model, 
                                    'intercept': result.params[0],
                                    'beta': result.params[1],
                                    'p_intercept': result.pvalues[0],
                                    'p_beta': result.pvalues[1],
                                    'r2': result.rsquared,
                                    'r2_adjust': result.rsquared_adj,
                                    'mean_abs_resid_norm': np.mean(abs(result.resid)) / np.mean(abs(records_det.density_mean)),
                                    'n_obs': result.nobs,
                                    'all_res': result
                            }, index = [0])
    else:
    
        model = model + ' -1'    
        result = sm.ols(formula = model, data = records_det).fit()

        regress_summary = pd.DataFrame({'experiment': experimental_code, 
                                    'exp_round': experimental_rounds[rn], 
                                    'model': model, 
                                    'intercept': np.nan,
                                    'beta': result.params[0],
                                    'p_intercept': np.nan,
                                    'p_beta': result.pvalues[0],
                                    'r2': result.rsquared,
                                    'r2_adjust': result.rsquared_adj,
                                    'mean_abs_resid_norm': np.mean(abs(result.resid)) / np.mean(abs(records_det.density_mean)),
                                    'n_obs': result.nobs,
                                    'all_res': result
                            }, index = [0])        
        
    #print(result.summary())
    
    return regress_summary



#-----------------------------------#
#            DEFINITIONS            # 
#-----------------------------------#

data_folder = r'C:\Users\admin\Documents\MyDoc\Aktivity\A_pedestrian_dimension\Pedestrian_dimension_code\Data_out_of_git'

experimental_code = r'\bottleneck'
experimental_rounds = ['_25ped_1', '_25ped_2','_25ped_3','_43ped_1','_43ped_2','_43ped_3','_43ped_4','_43ped_5']

detector = [0,150,-75,75]       # x_1, x_2, y_1, y_2
attractor = [0,0]               # x,y
vedge_angle = np.pi/3           # 60 degree left and right

regress_summary_all = pd.DataFrame()

       
#-----------------------------------#
#            CALCULATIONS           # 
#-----------------------------------#


rep = range(len(experimental_rounds))
 
for rn in rep:

    #  DATA PREPARATION 
    full_file_name = data_folder + experimental_code + experimental_rounds[rn] + '.csv'
    positions = pd.read_csv(full_file_name)
    positions.rename(columns = {'time(s)': 'time', 'x(cm)': 'x', 'y(cm)': 'y', 'density(m-2)': 'density'}, inplace = True)
    positions['angle'] = np.arctan((positions.x - attractor[0])/(positions.y - attractor[1]))                                       # angle toward the centre of exit
    positions['a_dist'] = np.sqrt(pow((positions.x - attractor[0]), 2) + pow((positions.y - attractor[1]), 2))                     # distance toward the centre of exit

    # TO DO: CALCULATE VORONOI DENSITY 
                                     
    positions_nearest = calculate_min_dist_vedge(positions)                                               

    records_det = calculate_detector_means(positions_nearest, positions, detector)

    regress_summary = run_regression(records_det, model = 'density_mean ~ mindist_inv_mean', intercept = True)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True)    
    regress_summary = run_regression(records_det, model = 'density_mean ~ mindist_inv2_mean', intercept = True)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True)  
    regress_summary = run_regression(records_det, model = 'density_mean ~ mindist_mean_inv', intercept = True)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True)  
    regress_summary = run_regression(records_det, model = 'density_mean ~ mindist_mean_inv2', intercept = True)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True)  
    regress_summary = run_regression(records_det, model = 'density_mean ~ mindist_inv_mean', intercept = False)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True)    
    regress_summary = run_regression(records_det, model = 'density_mean ~ mindist_inv2_mean', intercept = False)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True)  
    regress_summary = run_regression(records_det, model = 'density_mean ~ mindist_mean_inv', intercept = False)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True)  
    regress_summary = run_regression(records_det, model = 'density_mean ~ mindist_mean_inv2', intercept = False)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True)  
    
    
    regress_summary = run_regression(records_det, model = 'density_mean_log ~ mindist_mean_log', intercept = True)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True)  
    regress_summary = run_regression(records_det, model = 'density_mean_log ~ mindist_log_mean', intercept = True)
    regress_summary_all = regress_summary_all.append(regress_summary, ignore_index = True) 




#-----------------------------------#
#               PLOTS               # 
#-----------------------------------#



experimental_round = '_43ped_5'


plt.scatter(1/records_det.mindist_inv_mean, records_det.density_mean, label = 'density vs mindist: detector')

model_mindist = np.arange(30, 170, 1)

model = 'density_mean ~ mindist_inv2_mean'

beta = regress_summary_all[(regress_summary_all.exp_round == experimental_round) 
                        & (regress_summary_all.model == model)].beta
beta.reset_index(inplace = True, drop = True)
beta = beta[0]
intercept = regress_summary_all[(regress_summary_all.exp_round == experimental_round) 
                        & (regress_summary_all.model == model)].intercept
intercept.reset_index(inplace = True, drop = True)
intercept = intercept[0]

model_density = intercept + beta*(1/pow(model_mindist,2))

plt.plot(model_mindist, model_density, 'r-', label = model)



model = 'density_mean ~ mindist_inv_mean'

beta = regress_summary_all[(regress_summary_all.exp_round == experimental_round) 
                        & (regress_summary_all.model == model)].beta
beta.reset_index(inplace = True, drop = True)
beta = beta[0]
intercept = regress_summary_all[(regress_summary_all.exp_round == experimental_round) 
                        & (regress_summary_all.model == model)].intercept
intercept.reset_index(inplace = True, drop = True)
intercept = intercept[0]

model_density = intercept + beta*(1/model_mindist)

plt.plot(model_mindist, model_density, 'g-', label = model)


plt.legend()
plt.show()









# voronoi? https://pymesh.readthedocs.io/en/latest/api_mesh_generation.html

# příklad meshgrid
a = np.arange(-10, 10, 0.1)  
b = np.arange(-10, 10, 0.1)  
xa, xb = np.meshgrid(a, b, sparse=True)  
z = np.sin(xa**2 + xb**2) / (xa**2 + xb**2)  
h = plt.contourf(a,b,z)  
plt.show()  




















