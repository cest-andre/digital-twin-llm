import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SensorNet(nn.Module):
    def __init__(self, is_aws):
        super().__init__()

        self.net = None
        if is_aws:
            self.net = nn.Sequential(
                nn.Linear(451, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 3) # predict temp, humidity, and dewpoint
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(401, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 2) # predict temp and humidity
            )


    def forward(self, x):
        return self.net(x)


#   Loads input and output npy arrays, splits into train/val (80/20), optionally normalizes all features.
def load_data(sensor_id, norm=True, split=80):
    inputs = np.load(f"/media/andrelongon/DATA/sensor_data/{sensor_id}/inputs.npy")
    outputs = np.load(f"/media/andrelongon/DATA/sensor_data/{sensor_id}/outputs.npy")

    if norm:
        inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)


    num_train = int(inputs.shape[0] * (split/100))

    train_inputs = inputs[:num_train]
    train_outputs = outputs[:num_train]

    test_inputs = inputs[num_train:]
    test_outputs = outputs[num_train:]

    return train_inputs, train_outputs, test_inputs, test_outputs


def train_sensor_model(train_inputs, train_outputs, test_inputs, test_outputs, model, sensor_id, epochs=100, device="cuda"):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.RAdam(model.parameters())

    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_inputs), torch.Tensor(train_outputs))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)

    for ep in range(0, epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'[{ep + 1}, {i + 1:5d}] loss: {running_loss / i:.3f}')

        if ep != 0 and (ep+1) % 25 == 0:
            torch.save(model.state_dict(), f"/media/andrelongon/DATA/sensor_weights/{sensor_id}/model_1_{1+ep}ep.pth")
            torch.save(optimizer.state_dict(), f"/media/andrelongon/DATA/sensor_opt_states/{sensor_id}/model_1_{1+ep}ep.pth")

        #  validation
        val_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_inputs), torch.Tensor(test_outputs))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=False, drop_last=False)

        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                total += loss.item()

        print(f'Validation loss: {total:.3f}')


if __name__ == '__main__':
    aws_sensors = ['12345678-1234-1234-1234-123456789012']
    thingworx_sensors = ['ENV-110B20', 'ENV-110B27', 'ENV-110B85']

    for sensor_id in aws_sensors:
        model = SensorNet(True).to("cuda")

        train_inputs, train_outputs, test_inputs, test_outputs = load_data(sensor_id)
        train_sensor_model(train_inputs, train_outputs, test_inputs, test_outputs, model, sensor_id)

    for sensor_id in thingworx_sensors:
        model = SensorNet(False).to("cuda")

        train_inputs, train_outputs, test_inputs, test_outputs = load_data(sensor_id)
        train_sensor_model(train_inputs, train_outputs, test_inputs, test_outputs, model, sensor_id)