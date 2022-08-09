#计算climatology for validation for nci

# from scipy.stats import norm
# ps.crps_ensemble(obs,ens).shape
import sys
sys.path.append('../')
import util.data_processing_tool as dpt

import numpy as np
import properscoring as ps
from datetime import timedelta, date, datetime
import os
from os import mkdir


def rmse(ens,hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.sqrt( ((ens-hr)**2)  .sum(axis=(0))/ ens.shape[0] )

def mae(ens,hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens-hr)).sum(axis=0)/ ens.shape[0]



def main(year,time_windows):
    dates_needs=dpt.date_range(date(year, 1, 1),date(year+1, 1, 1))
    file_BARRA_dir="/scratch/iu60/yz9299/grid_05/high_agcd/"
#     date_map=np.array(dates_needs)
    # np.where(date_map==date(2012, 1, 1))
    crps_ref=[]
    mae_ref=[]
    rmse_ref=[]
    import tqdm
    for target_date in tqdm.tqdm(dates_needs):
        hr=dpt.read_agcd_data_fc(file_BARRA_dir,target_date)*51
        ensamble=[]
#         for y in range(1990,target_date.year):
        for y in range(1990,2012):

            if target_date.year==y:
                continue

            for w in range(1,time_windows): #for what time of windows

                filename=file_BARRA_dir+str(y)+(target_date-timedelta(w)).strftime("-%m-%d")+".nc"
                print(filename)
                if os.path.exists(filename):
                    t=date(y,(target_date-timedelta(w)).month,(target_date-timedelta(w)).day)
                    sr=dpt.read_agcd_data_fc(file_BARRA_dir,t)*51
                    ensamble.append(sr)

                filename=file_BARRA_dir+str(y)+(target_date+timedelta(w)).strftime("-%m-%d")+".nc"
                if os.path.exists(filename):

                    t=date(y,(target_date+timedelta(w)).month,(target_date+timedelta(w)).day)


                    sr=dpt.read_agcd_data_fc(file_BARRA_dir,t)*51
                    ensamble.append(sr)

            filename=file_BARRA_dir+str(y)+target_date.strftime("-%m-%d")+".nc"
           # print(filename)
            if os.path.exists(filename):
                t=date(y,target_date.month,target_date.day)
                sr=dpt.read_agcd_data_fc(file_BARRA_dir,t)*51
                ensamble.append(sr)
        #if ensamble:
        ensamble=np.array(ensamble)
        print(hr.shape,ensamble.shape)

        a=ps.crps_ensemble(np.squeeze(hr),np.squeeze(ensamble).transpose(1,2,0))
    #         a=vectcrps_m(ensamble,hr)
            #rmse_score=rmse(ensamble,hr)
            #mae_score=mae(ensamble,hr)


            #mae_ref.append(mae_score)
            #rmse_ref.append(rmse_score)
        crps_ref.append(a)

    #if not os.path.exists('./save/mae/climatology/'):
        #mkdir('./save/mae/climatology/')

    #if not os.path.exists('./save/rmse/climatology/'):
        #mkdir('./save/rmse/climatology/')

    if not os.path.exists('./save/crps/climatology_1/'):
        mkdir('./save/crps/climatology_1/')

    np.save("./save/crps/climatology/climatology_"+str(year)+"_all_lead_time_windows_"+str((time_windows-1)*2+1),np.array(crps_ref))
    #np.save("./save/mae/climatology/climatology_"+str(year)+"_all_lead_time_windows_"+str((time_windows-1)*2+1),np.array(mae_ref))
    #np.save("./save/rmse/climatology/climatology_"+str(year)+"_all_lead_time_windows_"+str((time_windows-1)*2+1),np.array(rmse_ref))

    print(year,time_windows)

if __name__=='__main__':
    #year_list=[1997,2010,2012]
    year_list=[2012]
    #timewind=[6]
    timewind=[1]
    for i in timewind:
        for j in year_list:
            main(j,i)