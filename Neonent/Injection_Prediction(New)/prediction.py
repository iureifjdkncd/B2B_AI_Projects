model_base_path = 'business_logic/models'

import os 
import re
import glob 
import joblib
import pickle
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from numpy import array
from tqdm import tqdm
import datetime
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
import pytz
from zoneinfo import ZoneInfo
import random
import tensorflow 
import tensorflow as tf
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import warnings 
warnings.filterwarnings(action='ignore')

def get_mongo_data(df,machine_number,time):
    server_address = "private_server_address"
    client_update = MongoClient("mongodb://private_compnay_client@{0}/admin".format(server_address))
    db = client_update["private_DB"] # db 이름 변경 
    col = db['productionData']
    utc_now = datetime.now(ZoneInfo("UTC"))
    if time=='UTC':
        from_date = datetime(utc_now.year, utc_now.month, utc_now.day, tzinfo=ZoneInfo("UTC"))
    elif time=='KST':
        kst_now = utc_now.astimezone(ZoneInfo("Asia/Seoul"))
        from_date = datetime(kst_now.year, kst_now.month, kst_now.day,tzinfo=timezone(timedelta(hours=9)))
    if df =='injection':
        df = pd.DataFrame(list(col.find({"MECHCD": f"A01-0{str(machine_number)}",
                                "$and":[{"TimeStamp":{"$gte":from_date}}]}).sort('TimeStamp',-1).limit(2500))).reset_index(drop=True)
        df = df.sort_values(by='TimeStamp').reset_index(drop=True)
        df[['Value1','Value2','Value3','Value4']] = df[['Value1','Value2','Value3','Value4']].astype(float)
        df['TimeStamp'] = df['TimeStamp'] + timedelta(hours=9)
        df.replace(to_replace=[None], value=np.nan, inplace=True)
        #### 1.) 불량 유형 명시된 cavity에는 최초 라벨링 0 부여 --> 1로 변환 
        df_defect_defined = df[df['LabelingDefectTypes'].notna()]
        df_defect_defined['passorfail'].fillna('0',inplace=True)
        df_defect_defined['passorfail'].replace({'0':'1'},inplace=True)
        df_defect_defined['passorfail'] = 1
        #### 2.) 불량유형 명시 안된 cavity에는 최초 라벨링 1 부여 --> 0으로 변환 
        df_defect_undefined = df[df['LabelingDefectTypes'].isna()]
        df_defect_undefined['passorfail'].fillna('1',inplace=True)
        df_defect_undefined['passorfail'].replace({'1':'0'},inplace=True)
        df_defect_undefined['LabelingDefectTypes'].fillna('No Defect',inplace=True)
        df_defect_undefined['passorfail'] = 0
        #### 3.) 라벨링 부여된 데이터 재정렬 
        df_concat = pd.concat([df_defect_defined,df_defect_undefined],axis=0).sort_values(by='TimeStamp').reset_index(drop=True)
        df_concat['passorfail'] = df_concat['passorfail'].astype(int)
        df_concat.dropna(axis=1,inplace=True)
        #### 4.) TimeStamp KST 표준화 --> (KST = UTC + 9)
        df_cavity = df_concat
        df_cavity = df_cavity.rename(columns={'Working_No':'UniqeNumAndCavity','passorfail':'PassOrFail','TimeStamp':'timeStamp'})
        return df_cavity

def get_work_shift(timestamp):
    hour = timestamp.hour
    if 0 <= hour < 6:
        return 'dawn'
    elif 6 <= hour < 9:
        return 'start_shift'
    elif 9 <= hour < 12:
        return 'day_shift1'
    elif 12 <= hour < 13:
        return 'lunch_break'
    elif 13 <= hour < 17:
        return 'day_shift2'
    elif 17 <= hour <19:
        return 'change_shift'
    else:
        return 'night_shift' 
        
def make_prediction_data(machine_number):
    try:
        df_cavity_kst = get_mongo_data('injection',machine_number,'KST')
        df_cavity_utc = get_mongo_data('injection',machine_number,'UTC')
        if len(df_cavity_kst) > len(df_cavity_utc):
            df_cavity = df_cavity_kst
        else:
            df_cavity = df_cavity_utc
    except:
        df_cavity = get_mongo_data('injection',machine_number,'UTC')
    df_cavity['time_slot'] = df_cavity.timeStamp.apply(get_work_shift)
    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    current_slot = get_work_shift(now_kst)
    df_cavity = df_cavity[df_cavity['time_slot'] == current_slot].reset_index(drop=True)
    print(df_cavity.timeStamp.iloc[0],'~',df_cavity.timeStamp.iloc[-1])
    shot_cols = ['Value1','Value2','Value3','Value4']
    df_inputs = df_cavity[['Working_No_Origin']+['UniqeNumAndCavity']+shot_cols+['PassOrFail']]#.tail(1).reset_index(drop=True)
    return df_inputs

def realtime_mae_loss_dist(test_data,machine_number):
    df_inputs = test_data
    shot_cols = ['Value1','Value2','Value3','Value4']
    df_recent_records = df_inputs[shot_cols]
    scaler_update = MinMaxScaler(feature_range=(0,1),clip=True)
    scaler_update.fit(df_recent_records) 
    X_recents = pd.DataFrame(scaler_update.transform(df_recent_records),columns =df_recent_records.columns)
    model = load_model(f'{model_base_path}/models/A01-0'+str(machine_number)+'_autoencoder_best_weight.h5')
    recent_preds = model.predict(X_recents)
    recent_mae_loss = np.mean(np.abs(recent_preds - X_recents), axis=1)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(recent_mae_loss)
        kde_samples = kde.resample(10000).flatten() 
        cv = (kde_samples.std() / kde_samples.mean())*100
    except:
        kde_samples = 0
        cv = 999
    return kde_samples, cv

def evaluate_distribution_stability(kde_samples, cv_threshold=60, skew_threshold=1.0, kurt_range=(2, 5)):
    from scipy.stats import skew, kurtosis, gaussian_kde
    try:
        mean = np.mean(kde_samples)
        std = np.std(kde_samples)
        cv = (std / mean) * 100 if mean > 1e-8 else 999
        skew_val = abs(skew(kde_samples))
        kurt_val = kurtosis(kde_samples, fisher=False)
        if (cv < cv_threshold and skew_val < skew_threshold and kurt_range[0] <= kurt_val <= kurt_range[1]):
            return True
        else:
            return False
    except:
        return False

def fixed_prediction(test_data,machine_number,UniqeNumAndCavity):
    df_load = test_data
    working_no_origin = UniqeNumAndCavity.split('-')[0]
    print('Working_No_Origin ==>',working_no_origin)
    df_load = df_load[df_load['Working_No_Origin']==str(working_no_origin)].iloc[-1:].reset_index(drop=True)
    df_inputs = df_load.drop(['Working_No_Origin','UniqeNumAndCavity','PassOrFail'],axis=1)
    if machine_number==13:
        fault_threshold = 0.06
    elif machine_number==14:
        fault_threshold = 0.36
    elif machine_number==16:
        fault_threshold = 0.23
    elif machine_number==17:
        fault_threshold = 0.07
    elif machine_number==18:
        fault_threshold = 0.23
    else:
        pass
    def load_npy_in_batches(file_path, batch_size=1000):
        full_data = np.load(file_path)
        total_samples = full_data.shape[0]
        for i in range(0, total_samples, batch_size):
            batch = full_data[i:i+batch_size]
            yield batch
    accumulated_trained_npy = []
    file_path = f'{model_base_path}/train_npy/Train_machine'+str(machine_number)+'.npy'
    for batch in load_npy_in_batches(file_path,batch_size=1000):
        accumulated_trained_npy.append(batch)
    trained_normal = np.vstack(accumulated_trained_npy)
    scaler = MinMaxScaler(feature_range=(0,1),clip=True)
    scaler.fit(trained_normal) 
    pred_inputs = df_inputs
    x_test = pd.DataFrame(scaler.transform(pred_inputs),columns = pred_inputs.columns)
    model = load_model(f'{model_base_path}/models/A01-0'+str(machine_number)+'_autoencoder_best_weight.h5')
    pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(pred - x_test), axis=1)
    test = pd.DataFrame(pred_inputs[:])
    test['mae_loss'] = test_mae_loss 
    test['threshold'] = fault_threshold
    test['anomaly'] = test['mae_loss'] > test['threshold']
    test['prediction'] = test["anomaly"].astype(int)
    result = test['prediction'].replace({0:1,1:0}).iloc[-1]
    if result==1:
        print('Update Trained Normal Data')
        updated_trained_data = np.vstack([trained_normal, pred_inputs])
        np.save(f'{model_base_path}/train_npy/Train_machine'+str(machine_number)+'.npy',updated_trained_data)
    else:
        pass
    print('Run Fixed Prediction')
    print('fixed threshold',fault_threshold)
    print('trained scaled mae loss',test_mae_loss[0].round(3))   
    return result

def adaptable_prediction(test_data,machine_number,UniqeNumAndCavity):
    df_inputs = test_data
    shot_cols = ['Value1','Value2','Value3','Value4']
    df_recent_records = df_inputs[shot_cols]
    scaler_update = MinMaxScaler(feature_range=(0,1),clip=True)
    scaler_update.fit(df_recent_records) 
    X_recents = pd.DataFrame(scaler_update.transform(df_recent_records),columns =df_recent_records.columns)
    model = load_model(f'{model_base_path}/models/A01-0'+str(machine_number)+'_autoencoder_best_weight.h5')
    recent_preds = model.predict(X_recents)
    recent_mae_loss = np.mean(np.abs(recent_preds - X_recents), axis=1)
    test_recents = pd.DataFrame(df_inputs[:])
    test_recents['mae_loss'] = recent_mae_loss 
    try:
        from scipy.stats import skew, kurtosis, gaussian_kde
        kde = gaussian_kde(recent_mae_loss)
        kde_samples = kde.resample(10000).flatten() 
        cv_score = min((kde_samples.std() / kde_samples.mean())*100 / 60, 1.0)
        skew_score = min(abs(skew(kde_samples)) / 1.0, 1.0)
        kurt_score = min(abs(kurtosis(kde_samples, fisher=False) - 3) / 2.0, 1.0)
        distribution_risk = (cv_score + skew_score + kurt_score) / 3
        if skew_score >= 0.7 or kurt_score >= 1.0:
            q = 99.9 - distribution_risk * (99.9 - 95)
        else:
            q = 99.7 - distribution_risk * (99.7 - 95)
    except:
        kde_samples = recent_mae_loss
        q = 99.7
    print('Optimized Threshold Percentile ==>', q)
    test_recents['threshold'] = np.percentile(kde_samples,q)
    test_recents['anomaly'] = test_recents['mae_loss'] > test_recents['threshold']
    test_recents['prediction'] = test_recents["anomaly"].astype(int)
    test_recents['prediction'] = test_recents['prediction'].replace({0:1,1:0})
    print(test_recents['prediction'].value_counts())
    recent_realtime_results = test_recents.tail(20)['prediction'].tolist()
    working_no_origin = UniqeNumAndCavity.split('-')[0]
    current_pred = test_recents[test_recents['Working_No_Origin']==str(working_no_origin)]['prediction'].iloc[-1]
    print('Working_No_Origin ==>',working_no_origin)
    print('adaptable threshold',test_recents[test_recents['UniqeNumAndCavity']==UniqeNumAndCavity]['threshold'].iloc[-1].round(3))
    print('real time scaled mae loss',test_recents[test_recents['UniqeNumAndCavity']==UniqeNumAndCavity]['mae_loss'].iloc[-1].round(3))  
    result = current_pred
    if result == 1:
        print('Prediction Result = Normal')
    else :
        print(' Prediction Result = Fault')
    return test_recents , result , recent_realtime_results

def recheck_anomalies(test_recents,UniqeNumAndCavity):
    mu = np.mean( test_recents['mae_loss'])
    sigma = np.std( test_recents['mae_loss'])
    USL = test_recents['threshold'].iloc[-1]
    LSL = 0.0
    cp = (USL-LSL)/(6*sigma)
    cpk = min((USL - mu) / (3 * sigma), (mu - LSL) / (3 * sigma))
    print(f"공정능력지수 ==> cp:{cp:.3f},cpk: {cpk:.3f}")
    predicted_normals = test_recents[test_recents['prediction'] == 1].copy()
    predicted_faults = test_recents[test_recents['prediction'] == 0].copy()
    if len(predicted_faults)!=0:
        print('Recheck Anomaly Preds')
        from scipy.stats import median_abs_deviation
        predicted_faults['robust_z_score'] = (predicted_faults['mae_loss'] -np.median(test_recents['mae_loss'])) / (median_abs_deviation(test_recents['mae_loss'])*1.4826)
        predicted_faults['mae_diff'] = predicted_faults['mae_loss'] - predicted_faults['threshold']
        def refined_label_A(row,cp,cpk):
            if cp < 1.0 or cpk < 1.0:
                if row['mae_diff'] <= 0.01:
                    return 1
                elif row['robust_z_score'] < 1.5 and row['mae_diff'] < 0.02:
                    return 1
                else:
                    return 0
            elif 1.0 <= cp < 1.33 and 1.0 <= cpk < 1.33:
                if row['mae_diff'] <= 0.015:
                    return 1
                elif row['robust_z_score'] < 2.0 and row['mae_diff'] < 0.03:
                    return 1
                else:
                    return 0
            else:
                if row['mae_diff'] <= 0.02:
                    return 1
                elif row['robust_z_score'] < 2.5 and row['mae_diff'] < 0.035:
                    return 1
                else:
                    return 0
        def refined_label_B(row,cp,cpk):
            def dynamic_criteria(cp, cpk):
                stability = min(cp, cpk)
                mae_tol = 0.035 - (stability * 0.015)  
                z_tol = 2.5 - (stability * 1.0)        
                return mae_tol, z_tol
            mae_tol, z_tol = dynamic_criteria(cp, cpk)
            if row['mae_diff'] <= mae_tol:
                return 1
            elif row['robust_z_score'] < z_tol and row['mae_diff'] < mae_tol * 1.5:
                return 1
            else:
                return 0
        if len(test_recents) <200:
            refined_inspection = predicted_faults.apply(lambda row: refined_label_A(row,cp,cpk), axis=1)
        elif 200 <=len(test_recents) <1000:
            refined_inspection_A = predicted_faults.apply(lambda row: refined_label_A(row,cp,cpk), axis=1)
            refined_inspection_B = predicted_faults.apply(lambda row: refined_label_B(row,cp,cpk), axis=1)
            count_A = sum(refined_inspection_A.values)
            count_B = sum(refined_inspection_B.values)
            if count_A >= count_B:
                refined_inspection = refined_inspection_A
            else:
                refined_inspection = refined_inspection_B
        else: 
             refined_inspection = predicted_faults.apply(lambda row: refined_label_B(row,cp,cpk), axis=1)
        predicted_faults['prediction'] = refined_inspection
        predicted_faults = predicted_faults.drop(['robust_z_score','mae_diff'],axis=1)
        test_recents = pd.concat([predicted_normals,predicted_faults],axis=0).sort_values(by='Working_No_Origin',ascending=True).reset_index(drop=True)
    else:
        pass  
    print(test_recents['prediction'].value_counts())
    recent_realtime_results = test_recents.tail(20)['prediction'].tolist()
    working_no_origin = UniqeNumAndCavity.split('-')[0]
    current_pred = test_recents[test_recents['Working_No_Origin']==str(working_no_origin)]['prediction'].iloc[-1] 
    result = current_pred
    if result == 1:
        print('Prediction Result = Normal')
    else :
        print(' Prediction Result = Fault')
    return result , recent_realtime_results

def majority_vote_recent_segments(recent_realtime_results):
    ratio_10 = sum(recent_realtime_results[-10:]) / len(recent_realtime_results[-10:])
    judge_10 = 1 if ratio_10 >= 0.8 else 0
    ratio_15 = sum(recent_realtime_results[-15:]) / len(recent_realtime_results[-15:])
    judge_15 = 1 if ratio_15 >= 0.75 else 0
    ratio_20 = sum(recent_realtime_results[-20:]) / len(recent_realtime_results[-20:])
    judge_20 = 1 if ratio_20 >= 0.7 else 0
    total = judge_10 + judge_15 + judge_20
    final_result = 1 if total >= 2 else 0
    return final_result

def hybrid_prediction(test_data,machine_number,UniqeNumAndCavity):
    kde_samples, cv = realtime_mae_loss_dist(test_data,machine_number)
    is_stable = evaluate_distribution_stability(kde_samples, cv_threshold=60, skew_threshold=1.0, kurt_range=(2, 5))
    if not is_stable and len(test_data)>=200:
        kde_samples, cv = realtime_mae_loss_dist(test_data.tail(100),machine_number)
        is_stable = evaluate_distribution_stability(kde_samples, cv_threshold=60, skew_threshold=1.0, kurt_range=(2, 5))
    else:
        pass
    if is_stable==False:
        print('Distribution UnStable ==> Activate Fixed Anomaly Detection')
        res = fixed_prediction(test_data,machine_number,UniqeNumAndCavity)
    else:
        print('Distribution Stable ==> Activate Hybrid Anomaly Detection')
        result_A = fixed_prediction(test_data,machine_number,UniqeNumAndCavity)
        test_recents , result_B , recent_realtime_results = adaptable_prediction(test_data,machine_number,UniqeNumAndCavity)
        result_B , recent_realtime_results =  recheck_anomalies(test_recents,UniqeNumAndCavity)
        if (result_A==1) and (result_B==1): 
            res = 1
        elif (result_A==0) and (result_B==0):
            res = 0
        elif (result_A==1) and (result_B==0):
            res = majority_vote_recent_segments(recent_realtime_results)
        elif (result_A==0) and (result_B==1):
            res = 1
    return res

def PredictionNeonent(machine_number,UniqeNumAndCavity):
    try:
        if machine_number not in [13,14,16,17,18]:
            print('No trained Machine ==> Temporary Result')
            res = 1
        else:
            now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
            current_shift = get_work_shift(now_kst)
            if ((current_shift == 'dawn' and now_kst.hour == 0 and now_kst.minute < 15) or
                (current_shift == 'start_shift' and now_kst.hour == 6 and now_kst.minute < 15) or
                (current_shift == 'day_shift1' and now_kst.hour == 9 and now_kst.minute < 15) or
                (current_shift == 'day_shift2' and now_kst.hour == 13 and now_kst.minute < 15) or
                (current_shift == 'night_shift' and now_kst.hour == 19 and now_kst.minute < 15) or 
                current_shift in ['lunch_break', 'change_shift']):
                print("재가동 초기 구간 → Default Normal")
                res = 1  
            else:
                test_data = make_prediction_data(machine_number)
                res =  hybrid_prediction(test_data,machine_number,UniqeNumAndCavity)
        return {"prediction_result":str(res)}
    except:
        print('Untrackable Exception')
        res = 1
        return {"prediction_result":str(res)}
    

#machine_number = 14
#test_data = make_prediction_data(machine_number)
#UniqeNumAndCavity = test_data['UniqeNumAndCavity'].iloc[-1]
#PredictionNeonent(machine_number,UniqeNumAndCavity)