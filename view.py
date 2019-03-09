import torch
import matplotlib.pyplot as plt
def view(generator, params, model, number):
    for i in range(number):
        X, y = generator.generate(**params)
        pred_y, h = model(torch.Tensor([X]).double())
        h_ = h.detach().numpy().squeeze()
        print(i, pred_y.detach().numpy(), y)
        plt.subplot(1, number * 2, 2*i+1)
        plt.imshow(X, cmap='gray')
        plt.subplot(1, number * 2, 2*i+2)
        plt.imshow(h_ / h_.max(), cmap='jet')
    plt.show()

if __name__ == '__main__':
    from data_producer import MNISTDataProducer
    gen = MNISTDataProducer()
    from model import MNISTBaseLineModel
    model = MNISTBaseLineModel(size=3*28).double()
    model.load_state_dict(torch.load('models/base-model',map_location='cpu'))
    view(gen, {'grid_size': 3, 'target': 6}, model, number=3)


