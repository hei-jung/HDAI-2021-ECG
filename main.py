from test_modules import Test, print_scores, plot_roc_curve
from models.resnet import resnet34
import torch

if __name__ == '__main__':
    model = resnet34()
    model.load_state_dict(torch.load('./trained_params/epoch_100_model_train_1'))
    test = Test(model, data_path='./data,npy', label_path='./label.npy')
    y_target, y_predicted = test.predict()
    print_scores(y_target, y_predicted)
    plot_roc_curve(y_target, y_predicted)
