from test_modules import predict, print_scores, plot_roc_curve
from models.resnet import resnet34
import torch

if __name__ == '__main__':
    model = resnet34()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('./trained_params/epoch_100_model_train_1', map_location=torch.device(device)))
    y_target, y_predicted = predict(model, data_path='./valid_data.npy', label_path='./valid_label.npy')
    print_scores(y_target, y_predicted)
    plot_roc_curve(y_target, y_predicted)
