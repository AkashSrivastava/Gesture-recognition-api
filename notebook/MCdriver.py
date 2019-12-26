import pandas as pd
import os, sys
pd.set_option('display.max_columns', 500)
def get_data(folder_path):

    # get list with filenames in folder and throw away all non ncsv
    df_main=pd.DataFrame()
    dirs=os.listdir("/Users/julialiu/Documents/Semester9/Mobile/Assignment2/CSV")
    folder_path="/Users/julialiu/Documents/Semester9/Mobile/Assignment2/CSV"
    print(dirs)
    for i in range(len(dirs)):
        print(folder_path+"/"+dirs[i])
        files = [file_path for file_path in os.listdir(folder_path+"/"+dirs[i]) if file_path.endswith('.csv')]
        print(files)

        if len(files)<1:
            continue
        # init return df with first csv
        for x in range(len(files)):
            print (x);
            df = pd.read_csv(os.path.join(folder_path+"/"+dirs[i], files[x]), )
            df['class'] = dirs[i].upper()
            print(df)
            df_main=df_main.append(df)
        #
        # for file_path in files[1:]:
        #     print('compare: {}'.format(file_path))
        #     df_other = pd.read_csv(os.path.join(folder_path, file_path))
        #
        #     # only keep the animals that are in both frames
        #     df_other['class']=dirs[i]
        #     print(df_other)
        #     df_main.append(df_other)

    return df_main

if __name__ == '__main__':
    matched = get_data("./data")
    print(matched)
    matched.to_csv("dataset.csv")
