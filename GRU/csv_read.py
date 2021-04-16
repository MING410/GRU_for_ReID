import os
import csv
import numpy as np
data_list=[]
label_list=[]
train_data_list=[]   
test_data_list=[]
train_label_list=[]
test_label_list=[]
csv_file_list = []
test_label_list2=[]

path="./output0117/"
name_dic={"horie":0,"kawashima":1,"kogure":2,"komori":3,"lxy":4,"mei":5}#,"shn":6,"takahashi":7,"yoshigawa":8}
for name in ["horie","kawashima","kogure","komori","lxy","mei"]:#,"shn","takahashi","yoshigawa"]:
    csv_f_path = os.path.join(path,"{}/".format(name))
    csv_f_path1 = os.path.join(csv_f_path,"{}".format(name))
    for i in range(1,41):        
        csv_path2 = csv_f_path1+"{}.csv".format(i)
        if not os.path.exists(csv_path2):
                continue
        else:
            csv_file = open(csv_path2)
            csv_reader_lines = csv.reader(csv_file)  
            for one_line in csv_reader_lines:
                data_list.append(one_line)
                label_list.append(name_dic[name])

            if i%5 ==0:
                csv_file = open(csv_path2)
                csv_reader_lines = csv.reader(csv_file)  
                for one_line in csv_reader_lines:
                    test_data_list.append(one_line)    #将读取的csv分行数据按行存入列表‘date’中
                    test_label_list.append(name_dic[name])
                    # if len(test_data_list[0]) != 21:
                    #     print("len:",len(test_data_list[0]))

            else:
                
                csv_file = open(csv_path2)
                csv_reader_lines = csv.reader(csv_file)  
                for one_line in csv_reader_lines:
                    train_data_list.append(one_line)  
                    train_label_list.append(name_dic[name])
                
# for i in range(0,int(len(train_label_list))):
#     if len(train_data_list[i])!= 22:
#         print(i)                    
#print(type(np.array(train_data_list)))
test_data = np.array(train_data_list)
train_data = np.array(train_data_list)
train_label=np.array(train_label_list)
label = np.array(label_list)
data=np.array(data_list)
for i in range(0,len(train_label_list),28):
    test_label_list2.append(train_label_list[i])
    
test_label = np.array(test_label_list2)

print(label.shape)
print(data.shape)