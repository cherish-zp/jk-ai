import pandas as pd

#读取CSV文件
file_name1 = "train.csv" ;
file_name2 = "test.csv" ;
file_name3 = "validation.csv" ;

file_name4 = "new_train.csv" ;
file_name5 = "new_test.csv" ;
file_name6 = "new_validation.csv" ;



file_names = [file_name1,file_name2,file_name3 , file_name4 , file_name5,file_name6];

root_dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-data/news/"

for file_name in file_names:
    print(file_name + "==============================================start")
    df = pd.read_csv(root_dir + file_name)
    #统计每个类别的数据量
    category_counts = df["label"].value_counts()

    #统计每个类别的比值
    total_data = len(df)
    category_ratios = (category_counts / total_data) *100

    print(category_counts)
    print(category_ratios)
    print(file_name + "================================================end")