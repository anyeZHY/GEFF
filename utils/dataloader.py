import pandas as pd
import numpy as np
from PIL import Image

datapath = 'assets/MPIIFaceGaze/'
colomn = ['Face', 'Left', 'Right', '3DGaze', '2DGaze']

def convert_path(path):
    result = datapath + 'Image/'
    for char in path:
        if char == '\\':
            result += '/'
        else:
            result += char
    return result

def procees_data():
    data = pd.DataFrame(columns=colomn)
    for i_label in range(10):
        labelpath = datapath + 'Label/p' + str(i_label).zfill(2) + '.label'
        df = pd.read_table(labelpath, delimiter=' ')
        # df = df.head()
        df = df[colomn]
        # print(len(df))
        for i_pic in range(len(df)):
            for col in ['Face', 'Left', 'Right']:
                facepath = datapath + 'Image/' + df[col][i_pic].replace('\\','/')
                im = np.array(Image.open(facepath))
                df[col][i_pic] = im
            for col in ['3DGaze', '2DGaze']:
                arr = np.array(list(map(float, df[col][i_pic].split(','))))
                df[col][i_pic] = arr
        data = pd.concat([data, df])
    data.to_pickle('assets/MPIIFaceGazeData.csv')

if __name__ == '__main__':
    procees_data()
