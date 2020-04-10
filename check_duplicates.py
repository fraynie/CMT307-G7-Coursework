import hashlib, os
duplicates = []
hash_keys = dict()

DATA_DIR = 'C:\\Users\\ma000310\\source\\repos\\CMT307\\SEM2\\data\\Flickr'
CATEGORIES = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

for category in CATEGORIES:
    print('processing category: ', category)
    path = os.path.join(DATA_DIR, category)

    images = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    for index, filename in  enumerate(os.listdir(path)):  #listdir('.') = current directory
        #if os.path.isfile(filename):
        with open(os.path.join(path, filename), 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys: 
            hash_keys[filehash] = index
        else:
            duplicates.append((index,hash_keys[filehash]))

    print('duplicates:', len(duplicates))

    for index in duplicates:
        try:
            os.remove(images[index[0]])
        except Exception as e:
            print('Corrupt file: ', e)
            pass            