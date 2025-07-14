import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

#读取CSV文件
file_name1 = "train.csv" ;
file_name2 = "test.csv" ;
file_name3 = "validation.csv" ;
file_names = [file_name1,file_name2,file_name3];

root_dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-data/Weibo/"

for file_name in file_names:

    csv_file_path = root_dir + file_name
    df = pd.read_csv(csv_file_path)

    #定义重采样策略
    #如果想要过采样，使用RandomOverSampler
    #如果想要欠采样，使用RandomUnderSampler
    #我们在这里使用RandomUnderSampler进行欠采样
    #random_state控制随机数生成器的种子
    rus = RandomUnderSampler(sampling_strategy="auto",random_state=42)

    #将特征和标签分开
    X = df[["text"]]
    Y = df[["label"]]
    print(Y)

    #应用重采样
    X_resampled,Y_resampled = rus.fit_resample(X,Y)
    print(Y_resampled)
    #合并特征和标签，创建系的DataFrame
    df_resampled = pd.concat([X_resampled,Y_resampled],axis=1)

    print(df_resampled)

    #保存均衡数据到新的csv文件
    df_resampled.to_csv(root_dir + "new_" + file_name,index=False)