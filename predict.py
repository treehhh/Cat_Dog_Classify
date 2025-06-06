import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 加载模型
model = load_model('cat_dog_classify_model.keras')

# 测试集路径
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(150, 150),
    batch_size=40,
    class_mode='binary')

class_names = ['cat', 'dog']
correct_count = {'cat': 0, 'dog': 0}
wrong_count = {'cat': 0, 'dog': 0}
for i in range(len(test_generator)):
    batch_X, batch_y = test_generator[i]
    pred_batch_X = model.predict(batch_X)
    pred_batch_X = np.where(pred_batch_X > 0.5, 1, 0)
    for j in range(len(batch_y)):
        true_label = 'cat' if batch_y[j] == 0 else 'dog'
        pred_label = 'cat' if pred_batch_X[j] == 0 else 'dog'
        if true_label == pred_label:
            correct_count[true_label] += 1
        else:
            wrong_count[true_label] += 1

cat_right = correct_count['cat']
cat_wrong = wrong_count['cat']
dog_right = correct_count['dog']
dog_wrong = wrong_count['dog']

matrix = np.array([[cat_right,cat_wrong],[dog_wrong,dog_right]])
print('混淆矩阵：')
print(matrix)

loss,accuracy = model.evaluate(test_generator)
print(f'测试集损失值: {loss:.4f}, 准确率: {accuracy:.4f}')

Precision = cat_right/(cat_right+dog_wrong)
print(f'精确率: {Precision:.4f}')

recall = cat_right/(cat_right+cat_wrong)
print(f'召回率: {recall:.4f}')

F1 = 2 * Precision * recall / (Precision + recall)
print(f'F1值: {F1:.4f}')



