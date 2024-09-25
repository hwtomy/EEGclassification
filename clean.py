import os
import shutil

def clear_directory(folder_path):
 
    if os.path.exists(folder_path):
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'fail {file_path} reason: {e}')
    else:
        print(f'fold {folder_path} Not exit')