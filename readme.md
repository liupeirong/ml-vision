## Image classification sample

This sample demonstrates how to do image classification with the following approaches:
1. Use [Azure Custom Vision Service](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/home)
2. Train with traditional machine learning algorithms such as SVM
3. Train with Keras and Tensorflow

- [azure_custom_vision_score](azure_custom_vision_score.ipynb) uses the model you trained in Azure Custom Vision service to do predictions. 
- [process_images](process_images.ipynb) resizes and normalizes input images.
- [image_classification_traditional_ml](image_classification_traditional_ml.ipynb) trains the clasifier with traditional ML algorithms.
- [image_classification_tensorflow](image_classification_tensorflow.ipynb) uses Keras to the train the model.
- [train_deploy_azure_ml](train_deploy_azure_ml.ipynb) demonstrates how to train the model remotely with Azure Machine Learning Service, and use Azure Machine Learning Service to deploy the model as a web service. You would need to refactor your code in Jupyter notebook to Python scripts - 
  * To train remotely, refactor your code to a script like [train.py](train.py)
  * To deploy, implement a script [score.py](score.py) and put the dependencies in [env.yml](env.yml)
- [deploy_without_aml](deploy_without_aml.ipynb) reveals what's actually happening behind the scene when you deploy a service from a model.
