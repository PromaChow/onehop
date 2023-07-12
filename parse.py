import pandas as pd
from tqdm import tqdm


df = pd.read_csv('times_data.csv')


for j in tqdm(range(27)):
    path = '/home/iit/Desktop/rankSVM/Time/'+str(j+1)+'/java_sig_method_call.csv'
    filtered_df = df[df['bid']==j+1]


    # filtered_df.to_csv('check.csv',index=False)

    methods_df = pd.read_csv(path,delimiter=',')

    for i in tqdm(range(methods_df.shape[0])):
        cor_m1_format = methods_df.iloc[i][0]+'$'+methods_df.iloc[i][1]+methods_df.iloc[i][2]
        cor_m2_format = methods_df.iloc[i][3]+'$'+methods_df.iloc[i][4]+methods_df.iloc[i][5]
        cor_m3_format = methods_df.iloc[i][0].split('.')[-1]+'$'+methods_df.iloc[i][1]+methods_df.iloc[i][2]
        cor_m4_format = methods_df.iloc[i][3].split('.')[-1]+'$'+methods_df.iloc[i][4]+methods_df.iloc[i][5]


        count = 0
    
        for index, row in filtered_df.iterrows():
            if cor_m1_format in row['methodId']:
                count = 1
                count_2 = 0
                
                ep = row['ep']
                ef = row['ef']
                np = row['np']
                nf = row['nf']
                for index_2, row_2 in filtered_df.iterrows():
                
                    if cor_m2_format == row_2['methodId']:
                        
                        count_2 = 1
                        filtered_df.at[index,'ep']+= row_2['ep']
                        filtered_df.at[index,'ef']+= row_2['ef']
                        filtered_df.at[index,'np']+= row_2['np']
                        filtered_df.at[index,'nf']+= row_2['nf']

                        filtered_df.at[index_2,'ep']+= ep
                        filtered_df.at[index_2,'ef']+= ef
                        filtered_df.at[index_2,'np']+= np
                        filtered_df.at[index_2,'nf']+= nf

                    

                # if count_2 == 0 :
                #     for index_2, row_2 in filtered_df.iterrows():
                #         if cor_m4_format in row_2['methodId']:

                #             print(row_2)
                                
                #             filtered_df.at[index,'ep']+= row_2['ep']
                #             filtered_df.at[index,'ef']+= row_2['ef']
                #             filtered_df.at[index,'np']+= row_2['np']
                #             filtered_df.at[index,'nf']+= row_2['nf']

                #             filtered_df.at[index_2,'ep']+= ep
                #             filtered_df.at[index_2,'ef']+= ef
                #             filtered_df.at[index_2,'np']+= np
                #             filtered_df.at[index_2,'nf']+= nf

    filtered_df.to_csv('check'+str(j)+'.csv',index=True)



            


        






