import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from PIL import Image
import utils

from sklearn.externals import joblib
from azureml.core import Run

#########################################################################
##  Remote training setup
#########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data-store', type=str, dest='data_store', help='data store mounting point')
parser.add_argument('--process-images', type=str, dest='reprocess_images', help='if True, reprocess the images')
parser.add_argument('--target-image-size', type=int, dest='target_image_size', help='resize to the target')
parser.add_argument('--model-name', type=int, dest='model_name', help='model_name)
args = parser.parse_args()

datastore = args.data_store
if args.reprocess_images == 'True':
    reprocessimages = True
targetimagesize = args.target_image_size
model_name = args.model_name

src = os.path.join(datastore, '/path/to/src')
tgt = os.path.join(datastore, '/path/to/tgt')
training_params = 'reprocessimages:{}, size:{}'.format(reprocessimages, targetimagesize)
print(training_params)

run = Run.get_context()

#########################################################################
## training code - should be same with remote or local training
#########################################################################         
if reprocessimages:
    utils.processAllImages(src, tgt, targetimagesize)
    
imgs, labels = utils.load(tgt)
img_train, img_test, label_train, label_test = train_test_split(imgs, labels)

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(img_train, label_train)
accuracy = svm_model_linear.score(img_test, label_test)
print(accuracy)
predictions = svm_model_linear.predict(img_test)
cm = confusion_matrix(label_test, predictions)
print (cm)

#########################################################################
## log results and save model to Azure ML
#########################################################################         
run.log('training params:', training_params)
run.log('accuracy', np.float(accuracy))

os.makedirs('outputs', exist_ok=True)
# file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=svm_model_linear, filename='outputs/' + model_name)
