import os
import numpy as np
import pandas as pd
import re
import sys

# RE pattern
m4a_pattern = '.*.m4a'
mp4_pattern = '.*.mp4'

def create_file():
    global command_line_file
    command_line_file = open(os.path.join(os.path.curdir, 'command.sh'),'w+',encoding='utf8')
    print('Finish creating the command file under current dir!')
    command_line_file.write("#!/bin/bash \n")

def write_command():
    # Open the excel file using pandas
    excel_df = pd.read_excel(excel_path)
    for filename in os.listdir(folder_path):
        if re.search(mp4_pattern,filename):
            # Deal with only m4a file to avoid repetition
            filename_split = filename.split('_')
            re_name = filename_split[0]+'_'+filename_split[1]
            for index, row in excel_df.iterrows():
                row_split = row['文件名'].split('_')
                row_re_name = row_split[0]+'_'+row_split[1]
                if row_re_name == re_name:
                    start_time = row['开始录制时间']
                    for find_mp4 in os.listdir(folder_path):
                        if re.search(row_re_name,find_mp4) and re.search(m4a_pattern,find_mp4):
                            out_put_file_name = row_re_name+'_人声'+'.wav'
                            command_string = './extra '+ os.path.join(ugc_m4a_path,filename)+' '+\
                                 os.path.join(ugc_m4a_path,find_mp4)+' '+str(start_time) +' '+\
                                 os.path.join(out_put_path,out_put_file_name)
                            command_line_file.write(command_string+'\n')
                            command_line_file.write('\n')

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('Please enter right arguements!')

    excel_path = sys.argv[1]
    folder_path = sys.argv[2]
    ugc_m4a_path = folder_path

    try:
        os.mkdir(os.path.join(os.path.curdir,'Extract_Result'))

    except OSError:
        print ("Creation of the directory Extract_Result failed")
        sys.exit()
    else:
        print ("Successfully created the directory Extract_Result")

    out_put_path = os.path.join(os.path.curdir,'Extract_Result')
    create_file()
    write_command()
    print('Finish writing the command line file!')

    # Finish writing and close the file
    command_line_file.close()
    print('Finish closing the file!')
