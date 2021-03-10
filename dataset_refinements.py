import os 

DATASET_PATH = 'E:\\IDAO2021\\track_1\\idao_dataset\\train\\'



def create_images_paths(dataset_path: str, file_name):
    
    names = os.listdir(dataset_path)
    f = open(file_name, 'w')
    for name in names: 
        f.write(dataset_path + name + '\n')
    
    f.close()


def create_labels(file_name: str, ds_file_name: str, is_er = False):
    # TODO de gatat pe lumina, regresie si clasificare in acelasi fisier
    f = open(file_name, 'r')
    ds = open(ds_file_name, 'w')
    try:
        while True:
            line = f.readline().strip()
            if line == None or line == "\n" or line == '':
                f.close()
                ds.close()
                return

            tok = line.split('\\')[-1]
            if is_er:
                reg_value = tok.split('_')[6]
                cls_value= '0'
            else:
                reg_value = tok.split('_')[7]
                cls_value= '1'
            ds.write(line.strip() + ',' + cls_value + ',' + reg_value + '\n')
            
    except:
        print(line)

        

        
        
        


# create_images_paths(DATASET_PATH + 'ER\\', 'ER_files.txt')
# create_images_paths(DATASET_PATH + 'NR\\', 'NR_files.txt')

create_labels('ER_files.txt', 'ER_dataset.txt', is_er=True)
create_labels('NR_files.txt', 'NR_dataset.txt')


