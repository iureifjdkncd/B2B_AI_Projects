import os 
import re
import glob 
import random
import joblib
import pickle
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from numpy import array
from pymongo import MongoClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import datetime
from datetime import datetime,timedelta
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow 
import tensorflow as tf
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
from tensorflow import keras
from tensorflow.keras import Input, Model,layers, models
from keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import warnings 
warnings.filterwarnings(action='ignore')

modelBasePath = 'business_logic/models'

def get_mongo_data(db_name,collection_name):
    server_address = "192.168.160.199:15115"
    client_update = MongoClient("mongodb://private_client@{0}/admin".format(server_address))
    db = client_update[db_name]
    col = db[collection_name]
    if collection_name == 'productiondata':
        df = pd.DataFrame(list(col.find({}).sort("TimeStamp", -1).limit(100)))
        df = df.sort_values(by='TimeStamp').reset_index(drop=True)
    elif collection_name == 'productionsettingdata':
        df = pd.DataFrame(list(col.find({}).sort("TimeStamp", -1).limit(100)))
        df = df.sort_values(by='TimeStamp').reset_index(drop=True)
    for col in df.columns:
        try : df[col] = df[col].astype('float64') # float32
        except: pass 
    return df

def get_productiondata():
    df_production = get_mongo_data('privateDB','productiondata')
    df_production['orderDate'] = df_production['orderNumber'].str.split('_').str[3]
    shot_cols = [col for col in df_production.columns if col.split('_')[0] == 'shotdata']
    df_settings = get_mongo_data('privateDB','productionsettingdata')
    df_settings.drop(['set_tc_s_f','set_tc_s_m','set_tm_s_f','set_tm_s_m'],axis=1,inplace=True)
    set_cols = [col for col in df_settings.columns if col.split('_')[0] == 'set']
    not_list = ['notFound','undefined','']
    df_settings = df_settings[df_settings['orderNumber'].isin(not_list) == False]
    df_settings = df_settings[df_settings.orderNumber.isin(not_list) == False]
    df_production = df_production[['_id','orderNumber','TimeStamp','Working_No','maker','moldNumber','moldName','MECHCD','MECHNM','SABUN','ITEMNM','orderDate','good_qty','bad_qty','prediction']+ shot_cols]
    df_setting = df_settings[['Working_No']+ set_cols]
    
    df_setting = df_setting.drop_duplicates(subset=['Working_No'])
    df_sensing = df_production.drop_duplicates(subset=['Working_No'])
    Production_Data = pd.merge(df_sensing,df_setting, how='inner',on=['Working_No']).reset_index(drop=True)
    Production_Data['_id'] = Production_Data['_id'].astype(str)
    Production_Data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    Production_Data = Production_Data.dropna(subset=['Working_No','maker','moldNumber','moldName','MECHCD','MECHNM','SABUN','ITEMNM'],axis=0).reset_index(drop=True)
    
    df_production_final = pd.merge(df_production,Production_Data[['Working_No']+set_cols],on='Working_No',how='left')
    df_production_final['MECHCD'] = df_production_final['MECHCD'].astype(str)
    df_production_final['maker'] = df_production_final['maker'].fillna(method='ffill').fillna(method='bfill')
    return df_production_final


def find_closest_trained_settings_info(dictionary,current_input,load_trained_settings,setting_cols,define_set_cluster,define_record,facility,item):
    if len(define_record)==0: 
        valid_clusters = dictionary[(dictionary['cluster_trainable'] == True)]['clusterSetting'].unique().tolist()
        change='False'
    else:
        valid_clusters = dictionary[(dictionary['MECHCD']==facility)&(dictionary['SABUN']==item) &
             (dictionary['cluster_trainable'] == True)]['clusterSetting'].unique().tolist()
        if len(valid_clusters)==0:
            valid_clusters = dictionary[(dictionary['cluster_trainable'] == True)]['clusterSetting'].unique().tolist()
            change='True'
        else:
            change='False'
            pass 
    setting_inputs = load_trained_settings[:, :-1]
    setting_clusters = load_trained_settings[:, -1]
    valid_mask = np.isin(setting_clusters, valid_clusters) & (setting_clusters != define_set_cluster)
    filtered_inputs = setting_inputs[valid_mask]
    filtered_clusters = setting_clusters[valid_mask]
    query = current_input[setting_cols].values
    if query.ndim == 1:
        query = query.reshape(1, -1)

    all_data = np.vstack([filtered_inputs, query])  # shape: (N+1, D)
    min_vals = np.min(all_data, axis=0)
    max_vals = np.max(all_data, axis=0)
    denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    scaled_all = (all_data - min_vals) / denom
    scaled_filtered_inputs = scaled_all[:-1]
    scaled_query = scaled_all[-1].reshape(1, -1)
    
    nearest_idx, distances = pairwise_distances_argmin_min(scaled_query, scaled_filtered_inputs)
    max_possible_dist = np.sqrt(scaled_query.shape[1])
    normalized_dist = distances[0] / max_possible_dist
    similarity_score = 1 - normalized_dist

    nearest_setting = filtered_inputs[nearest_idx[0]]
    nearest_cluster = filtered_clusters[nearest_idx[0]].astype(int)
    define_set_cluster = nearest_cluster
    if len(define_record)==0 or (change=='True'):
        #print('Change Direction')
        near_info = dictionary[(dictionary['cluster_trainable']==True)&(dictionary['clusterSetting']==define_set_cluster)].sample(1,random_state=2021)  
        facility = near_info['MECHCD'].iloc[-1]
        item  = near_info['SABUN'].iloc[-1]
        return facility, item, define_set_cluster,similarity_score
    else:
        return define_set_cluster,similarity_score


def calculate_adaptive_threshold(load_trained_model,select_model,load_trained_infos,test,adaptable,similarity_score):
    trained_data = load_trained_infos[:, :-1]
    trained_threshold = load_trained_infos[:, -1][0]
    if select_model == 'ISF':
        train_mae_loss = trained_data # trained_scores
    elif select_model == 'AE':
        scaler = MinMaxScaler(clip=True)
        scaler.fit(trained_data)
        train_scaled = pd.DataFrame(scaler.transform(trained_data), columns=test.columns)
        train_preds = load_trained_model.predict(train_scaled)
        train_mae_loss = np.mean(np.abs(train_preds - train_scaled), axis=1)
    elif select_model == 'memae':
        from trained.memae import memae_anomaly_scores, memae_anomaly_scores_extended
        scaler = MinMaxScaler(clip=True)
        scaler.fit(trained_data)
        train_scaled = pd.DataFrame(scaler.transform(trained_data), columns=test.columns)
        MemAE = load_trained_model
        train_mae_loss = memae_anomaly_scores_extended(train_scaled, MemAE)
    max_threshold = np.max(train_mae_loss)#np.percentile(train_mae_loss,99.9)
    final_threshold = max(trained_threshold,max_threshold)
    cv = np.abs(np.std(train_mae_loss) / np.mean(train_mae_loss))
    if adaptable=='True':
        if select_model=='ISF':
            train_mae_loss = train_mae_loss.reshape(-1)
        else:
            pass
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(train_mae_loss)
        kde_samples = kde.resample(100000).flatten()
        train_mae_loss = kde_samples
        cv = np.abs(np.std(train_mae_loss) / np.mean(train_mae_loss))
        if cv < 0.1:
            cv_alpha = np.random.uniform(0.012, 0.02)
        elif cv < 0.2:
            cv_alpha = np.random.uniform(0.025, 0.032)
        elif cv < 0.4:
            cv_alpha = np.random.uniform(0.04, 0.05)
        elif cv < 0.6:
            cv_alpha = np.random.uniform(0.06, 0.07)
        else:
            cv_alpha = np.random.uniform(0.08, 0.09)
        if similarity_score >= 0.8:
            adjustment = np.random.uniform(0.025, 0.03)
        elif similarity_score >= 0.6:
            adjustment = np.random.uniform(0.035, 0.04)
        elif similarity_score >= 0.4:
            adjustment = np.random.uniform(0.045, 0.052)
        elif similarity_score >= 0.2:
            adjustment = np.random.uniform(0.055, 0.065)
        else:
            adjustment = np.random.uniform(0.08, 0.095)
        adjustment = adjustment + (1 - similarity_score)**2 * 0.025 # 0.01 , 0.03 , 0.05 , 0.075 
        #cv_alpha += (cv - 0.3) * 0.02
        final_threshold = final_threshold + adjustment #+ cv_alpha

        #cv_norm = np.clip(cv / 0.6, 0, 1)                        
        #sim_norm = (1 - similarity_score)**2     
        #margin_factor = (cv_norm + sim_norm) / 2            
        #flexible_margin = 0.05 + margin_factor * 0.05 # ( 0.05 ~ 0.1 ) 
        margin_factor = (cv / 0.6 + (1 - similarity_score)**2) / 2
        flexible_margin = np.clip(margin_factor * 0.1, 0, 0.2) # + 0.05
    else:
        cv = np.abs(np.std(train_mae_loss) / np.mean(train_mae_loss))
        if cv < 0.05:
            flexible_margin = 0.005  # 거의 무변동한 경우
        elif cv < 0.1:
            flexible_margin = 0.01
        elif cv < 0.2:
            flexible_margin = 0.015
        elif cv < 0.4:
            flexible_margin = 0.02
        elif cv < 0.6:
            flexible_margin = 0.03
        else:
            flexible_margin = 0.04 
    return final_threshold , flexible_margin


def prediction(df,working_number):
    Production_Data = df
    current_input = Production_Data[Production_Data['Working_No']==working_number].reset_index(drop=True)
    if len(current_input)>1:
        current_input = Production_Data[Production_Data['Working_No']==working_number].sample(n=1).reset_index(drop=True)
    else:
        pass
    maker = current_input['maker'].values[0]
    try:
        facility = current_input['MECHCD'].astype(float).astype(int).astype(str).values[0]
    except:
        facility = current_input['MECHCD'].astype(str).values[0]
    item = current_input['SABUN'].values[0]
    dictionary = pd.read_csv(f"{modelBasePath}/dictionary/{maker}_dictionary.csv", encoding='cp949')
    dictionary['MECHCD'] = dictionary['MECHCD'].astype(str)
    
    sensing_cols = np.load(f"{modelBasePath}/columns/{maker}/shot_cols.npy")
    setting_cols = np.load(f"{modelBasePath}/columns/{maker}/setting_cols.npy")
    Production_Data[sensing_cols] = Production_Data[sensing_cols].fillna(method='ffill').fillna(method='bfill')
    Production_Data[setting_cols] = Production_Data[setting_cols].fillna(method='ffill').fillna(method='bfill')

    sensing_modes = pd.read_csv(f"{modelBasePath}/Mode_values/{maker}_sensing_dict.csv",encoding='cp949')
    loaded_sensing_dict = dict(zip(sensing_modes['column'], sensing_modes['mode_value']))
    setting_modes = pd.read_csv(f"{modelBasePath}/Mode_values/{maker}_set_dict.csv",encoding='cp949')
    loaded_setting_dict = dict(zip(setting_modes['column'], setting_modes['mode_value']))
    Production_Data[sensing_cols] = Production_Data[sensing_cols].apply(lambda col:col.fillna(loaded_sensing_dict.get(col.name,col)),axis=0)
    Production_Data[setting_cols] = Production_Data[setting_cols].apply(lambda col:col.fillna(loaded_setting_dict.get(col.name,col)),axis=0)
    #print(maker , facility, item)
    
    current_input = Production_Data[Production_Data['Working_No']==working_number].reset_index(drop=True)
    if len(current_input)>1:
        current_input = Production_Data[Production_Data['Working_No']==working_number].sample(n=1).reset_index(drop=True)
    else:
        pass
    check_sensings = current_input[sensing_cols].isna().sum().mean()
    check_settings = current_input[setting_cols].isna().sum().mean()
    if (check_sensings!=0) or (check_settings!=0):
        print('==> Input Nan Exception [Streaming Train] ')
        print('==> WorkingNumber', working_number)
        if current_input['bad_qty'].values[0]==0:
            result = 1
        else:
            result = 0
        print('==> prediction result :','Normal' if result==1  else 'Fault')
        print('='*150)
    else:
        try: # try
            print('==> WorkingNumber', working_number)
            load_kmeans = joblib.load(f"{modelBasePath}/cluster_settings/{maker}_settings.pkl")
            load_trained_settings = np.load(f"{modelBasePath}/setting_values/{maker}_settings.npy")
            define_set_cluster = load_kmeans.predict(current_input[setting_cols])[0]
            define_record = dictionary[(dictionary['MECHCD']==facility)&
                                       (dictionary['SABUN']==item)&
                                        (dictionary['clusterSetting']==define_set_cluster)].reset_index(drop=True)
            if len(define_record)==0:
                adaptable ='True'
                print('==> Search Closest Trained Setting from Dictionary')
                facility, item, define_set_cluster,similarity_score = find_closest_trained_settings_info(dictionary,current_input,load_trained_settings,setting_cols,
                                                                                        define_set_cluster,define_record,facility,item)
            else:
                check_cluster_available = define_record['cluster_trainable'].values[0]
                if check_cluster_available == False:
                    adaptable ='True'
                    print('==> Search Closest Trained Setting from current Facility/ItemCode')
                    results = find_closest_trained_settings_info(dictionary,current_input,load_trained_settings,setting_cols,
                                                        define_set_cluster,define_record,facility,item)
                    if len(results) == 2:
                        define_set_cluster, similarity_score = results
                    elif len(results) == 4:
                        facility, item, define_set_cluster, similarity_score = results
                else:
                    adaptable ='False'
                    similarity_score = 0
                    pass
            print('==> Setting Based Prediction')
            load_trained_infos = np.load(f"{modelBasePath}/inference_info/{maker}/Facility={facility}_ITEM={item}_setting={define_set_cluster}.npy")
            try:
                select_model='ISF'
                load_trained_model = joblib.load(f"{modelBasePath}/trained/{maker}/Facility={facility}_ITEM={item}_setting={define_set_cluster}.pkl")
            except:
                try:
                    select_model = 'AE'
                    load_trained_model = load_model(f"{modelBasePath}/trained/{maker}/Facility={facility}_ITEM={item}_setting={define_set_cluster}.h5")
                except:
                    select_model = 'memae'
                    from trained.memae import MemoryModule,memae_loss,l2_normalize,hard_shrinkage,entropy_loss,memae_anomaly_scores,memae_anomaly_scores_extended
                    custom_objects = {'MemoryModule': MemoryModule,'memae_loss': memae_loss,'l2_normalize': l2_normalize,
                                       'hard_shrinkage': hard_shrinkage,'entropy_loss': entropy_loss}
                    filename = f"{modelBasePath}/trained/{maker}/Facility={facility}_ITEM={item}_setting={define_set_cluster}_memae.h5"
                    load_trained_model = load_model(filename,custom_objects=custom_objects)
            test = current_input[sensing_cols]
            train = load_trained_infos[:, :-1]
            #threshold = load_trained_infos[:, -1][0]
            if select_model=='ISF':
                test_mae_loss = (-1 * load_trained_model.decision_function(test))[0] #scores_test = test_mae_loss
                threshold, margin = calculate_adaptive_threshold(load_trained_model,select_model,load_trained_infos,test,adaptable,similarity_score)
                add_df = pd.concat([pd.DataFrame(np.array([test_mae_loss])),pd.DataFrame(np.array([load_trained_infos[:, -1][0]]))],axis=1)
            elif select_model=='AE':
                scaler = MinMaxScaler(clip=True)
                threshold, margin = calculate_adaptive_threshold(load_trained_model,select_model,load_trained_infos,test,adaptable,similarity_score)
                scaler.fit(train)
                X_test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)
                x_test_pred = load_trained_model.predict(X_test_scaled)
                test_mae_loss = np.mean(np.abs(x_test_pred - X_test_scaled), axis=1)[0]
                add_df = pd.concat([test,pd.DataFrame(np.array([load_trained_infos[:, -1][0]]))],axis=1)
            elif select_model=='memae':
                scaler = MinMaxScaler(clip=True)
                threshold, margin = calculate_adaptive_threshold(load_trained_model,select_model,load_trained_infos,test,adaptable,similarity_score)
                scaler.fit(train)
                X_test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)
                from trained.memae import memae_anomaly_scores, memae_anomaly_scores_extended
                MemAE = load_trained_model 
                test_mae_loss = memae_anomaly_scores_extended(X_test_scaled, MemAE)[0]
                add_df = pd.concat([test,pd.DataFrame(np.array([load_trained_infos[:, -1][0]]))],axis=1)
            result = 1 if test_mae_loss < threshold else 0
            if result == 0:
                if test_mae_loss - threshold < margin: 
                    result = 1  
                else:
                    pass  
            else:
                pass  
            #if (result==1) & (adaptable=='False'):
                #update_info = np.vstack((load_trained_infos, np.array(add_df)))
                #np.save(f"{modelBasePath}/inference_info/{maker}/Facility={facility}_ITEM={item}_setting={define_set_cluster}.npy",update_info)
            #else:
            #    pass
            print('==> model',select_model)
            print(f"==> Maker={maker}_Facility={facility}_ITEM={item}" + (f"_setting={define_set_cluster}"))# 
            print('==> mae_recon_loss',test_mae_loss)
            print('==> threshold',threshold)
            print('==> margin limit',margin)
            print('==> prediction result :','Normal' if result==1  else 'Fault')
            print('='*150) 
        except:   
            print('==> No Trained Information (Current Quantity)')
            print('==> WorkingNumber', working_number)
            if current_input['bad_qty'].values[0]==0:
                result = 1
            else:
                result = 0
            print('='*150)
    return result


def Prediction(working_number):
    try:
        df = get_productiondata()
        res = prediction(df,working_number)
    except:
        res = -1
        print('Excpetion [Prediction Fail]')
        print('='*150)
    return {"prediction_result":str(res)}


df = get_productiondata()
working_number = str(df['Working_No'].iloc[-1])
Prediction(working_number)



#from business_logic.models.trained.memae --> code check from monitroing ipynb 

# http://127.0.0.1:25000/docs#/default/get_prediction_get_prediction_get




