if __name__ == '__main__':
    import os

    datasets = os.listdir('./')
    for data in datasets:
        os.system("montage -border 0 -geometry 660x -tile 3x2 ./{}/*.png ./montage/{}.png".format(data, data))
