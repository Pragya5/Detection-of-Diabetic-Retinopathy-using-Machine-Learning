import time
import pandas
import pickle
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

def prepareData():

 # Read in reduced data and make copies that will be mutated.
 full_PCA_data = pickle.load(open('reduced_data.bin', 'rb'), encoding='ISO8859-1')
 limited_PCA_data = full_PCA_data
 binary_PCA_data = full_PCA_data
 # Read in labels from CSV for training.
 df = pandas.read_csv('trainLabels.csv')
 temp_labels = df.as_matrix()
 full_labels = df.as_matrix()[:,1]
 limited_labels = full_labels
 binary_labels = full_labels
 # Creates 'limited' and 'binary' data which have fewer '0' class objects so
 # that the number of examples for each class is more balanced...'limited' has
 # a number of '0' class objects equal to the number of class '2' objects
 # while 'binary' has the same number of '0' class objects and additionally
 # removes all '1', '3', and '4' class objects.


 # Builds sets of row indices to be included in the final data subsets for
 # 'limited' and 'binary'.
 i, counter, limited, binary = 0, 0, set(), set()
 for i, l in enumerate(temp_labels):
 if counter < 5292 and l[1] == 0:
 limited.add(i)
 binary.add(i)
 counter += 1
 elif l[1] == 2:
 limited.add(i)
 binary.add(i)
 elif l[1] != 0:
 limited.add(i)
 # Removes row indices that do not belong in the 'limited' data/label subsets.
 i, j = 0, 0
 while len(limited_PCA_data) > len(limited):
 if j not in limited:

 limited_PCA_data = np.delete(limited_PCA_data, i, 0)
 limited_labels = np.delete(limited_labels, i, 0)
 i -= 1
 i += 1
 j += 1
 # Removes row indices that do not belong in the 'binary' data/label subsets.
 i, j = 0, 0
 while len(binary_PCA_data) > len(binary):
 if j not in binary:
 binary_PCA_data = np.delete(binary_PCA_data, i, 0)
 binary_labels = np.delete(binary_labels, i, 0)
 i -= 1
 i += 1
 j += 1
 # 'test_size' and 'seed' for splitting so that splits are the same for each
run.
 seed = 7
 size = 0.2

 # Data and their appropriate labels for each training configuration.
 configurations = [
 ('full', full_PCA_data, full_labels),
 #('limited', limited_PCA_data, limited_labels),
 #('binary', binary_PCA_data, binary_labels)
 ]
 # Initialized data dictionary that is returned by this funciton.
 data = {
 'full': {} #, 'limited': {}, 'binary': {}
 }
 # Populate data dictionary with appropriate data/label subsets.
 for c in configurations:
 X_train, X_test, y_train, y_test = train_test_split(*c[1:], test_size=size,
random_state=seed)
 data[c[0]]['X_train'] = X_train
 data[c[0]]['X_test'] = X_test
 data[c[0]]['y_train'] = y_train
 data[c[0]]['y_test'] = y_test

 # Convert labels from strings to ints.
 for _, v in data.items():
 v['y_train'], v['y_test'] = v['y_train'].astype(int),
v['y_test'].astype(int)
 return data


def trainModels(data, params):
 f = open(out_file, 'w')
 def trainKNN(data_subset):
 f.write('\nTraining KNN:'+'\n')
 X_train = data[data_subset]['X_train']
 X_test = data[data_subset]['X_test']
 y_train = data[data_subset]['y_train']
 y_test = data[data_subset]['y_test']
 for p in params['knn']:
 header = "@ subset: {0}, params: {1}".format(data_subset, p)
 f.write('\n'+header+'\n')
 n_neighbors = p['n_neighbors']
 model = KNeighborsClassifier(n_neighbors=n_neighbors)


 start = time.time()
 model.fit(X_train, y_train)
 elapsed_train = time.time() - start
 y_pred = model.predict(X_test).astype(int)
 elapsed_predict = time.time() - start
 accuracy = accuracy_score(y_test, y_pred)
 precision, recall, fscore, support =
precision_recall_fscore_support(y_test, y_pred, pos_label=2,
average='weighted')
 print("\n{5}\nKNN with {0} neighbors on data subset {1} trained in {2}
seconds and predicted in {3} seconds with an accuracy of
{4}\n".format(n_neighbors, data_subset, elapsed_train, elapsed_predict,
accuracy, header))
 f.write(str(elapsed_train) + ', ' + str(elapsed_predict) + str(accuracy)+
', ' + str(precision)+ ', ' + str(recall )+ ', ' + str(fscore )+ ', ' +
str(support))
 def trainSVM(data_subset):
 f.write('\nTraining SVM:'+'\n')
 X_train = data[data_subset]['X_train']
 X_test = data[data_subset]['X_test']


y_train = data[data_subset]['y_train']
 y_test = data[data_subset]['y_test']
 for p in params['svm']:
 header = "@ subset: {0}, params: {1}".format(data_subset, p)
 f.write('\n'+header+'\n')
 kernel = p['kernel']
 gamma = p['gamma']
 model = svm.SVC(kernel=kernel, gamma=gamma)
 start = time.time()
 model.fit(X_train, y_train)

elapsed_train = time.time() - start
 y_pred = model.predict(X_test).astype(int)
 elapsed_predict = time.time() - start
 accuracy = accuracy_score(y_test, y_pred)
 precision, recall, fscore, support =
precision_recall_fscore_support(y_test, y_pred, pos_label=2,
average='weighted')
 print("\n{5}\nSVM with {0} kernel and {6} gamma on data subset {1}
trained in {2} seconds and predicted in {3} seconds with an accuracy of
{4}\n".format(kernel, data_subset, elapsed_train, elapsed_predict, accuracy,
header, gamma))

 f.write(str(elapsed_train) + ', ' + str(elapsed_predict) + str(accuracy)+
', ' + str(precision)+ ', ' + str(recall )+ ', ' + str(fscore )+ ', ' +
str(support))
 # Iterate over all the data/label subsets and then train and predict with
 # each type of model.
 for k, v in data.items():
 print("\nTraining {0} subset on KNN classifier\n".format(k))
 trainKNN(k)
 print("\nTraining {0} subset on SVM classifier\n".format(k))
 trainSVM(k)
 f.close()
if __name__ == '__main__':
 # Dictionary of parameters keyed by the type of model. Values are lists of
 # parameter sets.
 params = {
 'knn': [{'n_neighbors': 3}, {'n_neighbors': 5},
 {'n_neighbors': 10}, {'n_neighbors': 25}, {'n_neighbors': 50},
{'n_neighbors':100}],
 'svm': [
 {'kernel': 'rbf', 'gamma': 0.1},
 {'kernel': 'rbf', 'gamma': 1.0},

 {'kernel': 'linear','gamma': 0.1},
{'kernel': 'linear','gamma': 1.0},
 {'kernel': 'poly', 'gamma': 0.1},
 {'kernel': 'poly', 'gamma': 1.0}
 ]
 }
 out_file = 'results.dat'
 start = time.time()
 data = prepareData()
 trainModels(data, params)
 print("\nEntire training took {0} seconds".format(time.time() - start))


# change this as you see fit
image_path = 'train/114_right.jpeg'
# Read in the image_data
image_data = tf.gfile.GFile(image_path, 'rb').read()
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
 in tf.gfile.GFile("tf_files/retrained_labels.txt")]
# Unpersists graph from file
with tf.gfile.GFile("tf_files/retrained_graph.pb", 'rb') as f:
 graph_def = tf.GraphDef()
 graph_def.ParseFromString(f.read())
 _ = tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
 # Feed the image_data as input to the graph and get first prediction
 softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
 predictions = sess.run(softmax_tensor,


{'DecodeJpeg/contents:0': image_data})
 # Sort to show labels of first prediction in order of confidence
 #top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
 #for node_id in top_k:
 human_string = label_lines[0]
 score = predictions[0][0]
print('%s (score = %.5f)' % (human_string, score))
human_string = label_lines[1]
score = predictions[0][1]
print('%s (score = %.5f)' % (human_string, score))
human_string = label_lines[2]
score = predictions[0][2]
print('%s (score = %.5f)' % (human_string, score))
human_string = label_lines[3]
score = predictions[0][3]
print('%s (score = %.5f)' % (human_string, score))
#human_string = label_lines[4]
#score = predictions[0][4]
#print('%s (score = %.5f)' % (human_string, score))

import cv2
import numpy as np
def adjust_gamma(image, gamma=1.0):

 table = np.array([((i / 255.0) ** gamma) * 255
 for i in np.arange(0, 256)]).astype("uint8")
 return cv2.LUT(image, table)
def extract_ma(image):
 r,g,b=cv2.split(image)
 comp=255-g
 clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
 histe=clahe.apply(comp)
 adjustImage = adjust_gamma(histe,gamma=3)
 comp = 255-adjustImage
 J = adjust_gamma(comp,gamma=4)
 J = 255-J
 J = adjust_gamma(J,gamma=4)

 K=np.ones((11,11),np.float32)
 L = cv2.filter2D(J,-1,K)

 ret3,thresh2 = cv2.threshold(L,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
 kernel2=np.ones((9,9),np.uint8)
 tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
 kernel3=np.ones((7,7),np.uint8)

 opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
 return opening
if __name__ == "__main__":

 import os, os.path
 from os.path import isfile, join
 path=os.getcwd()
 dirs = os.listdir(path)
 onlyfiles = [f for f in dirs if isfile(join(path, f))]
 onlyimages = [f for f in onlyfiles if f.endswith('.jpeg')]
 i = 0
 for item in onlyimages:
 fullpath = os.path.join(path,item)
#corrected
 if os.path.isfile(fullpath):
 fundus = cv2.imread(fullpath)
 f, e = os.path.splitext(fullpath)
 bloodvessel = extract_ma(fundus)
 cv2.imwrite(f+'_clahe.jpeg',bloodvessel)
 i+=1
 print("images left: "+str(len(onlyimages)-i))