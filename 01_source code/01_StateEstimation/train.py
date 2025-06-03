"""
created by
"""
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader.data_loader import load_data
from model.transformer import TransformerRegressor


parser = argparse.ArgumentParser(description='Train a multimodal model for state estimation of the soft gripper.')
parser.add_argument('--tactile_modal', type=bool, default=True, help='Whether to take tactile modal as input.')
parser.add_argument('--chamber_modal', type=bool, default=True, help='Whether to take chamber modal as input.')
parser.add_argument('--pressure_modal', type=bool, default=True, help='Whether to take pressure modal as input.')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--data_dir', type=str, default='./dataset/05_known_objects/', help='Dataset direction for loading.')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio.')
parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio.')
parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
parser.add_argument('--norm_max', type=float, default=0.15, help='Maximum value for finger state normalization.')
parser.add_argument('--norm_min', type=float, default=-0.15, help='Minimum value for finger state normalization.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--save_interval', type=int, default=1, help='Save interval of the weights as checkpoints.')
parser.add_argument('--save_dir', type=str, default='./checkpoints/7/', help='Directory to save checkpoints.')
args = parser.parse_args()


if torch.cuda.is_available():
    print("CUDA is available")
    print("Device name:", torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)


def main():

    train_loader, val_loader, test_loader = load_data(data_dir=args.data_dir, batch_size=args.batch_size,
                                                      train_ratio=args.train_ratio,
                                                      val_ratio=args.val_ratio,
                                                      test_ratio=args.test_ratio)

    model = TransformerRegressor(d_model=32, nhead=1, num_layers=1,
                                 tactile_modal=args.tactile_modal,
                                 chamber_modal=args.chamber_modal,
                                 pressure_modal=args.pressure_modal).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss_history = []
    val_loss_history = []
    average_distance_error_th_epoch_list, average_distance_error_ff_epoch_list, average_distance_error_mf_epoch_list = [], [], []

    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0
        for (inputs_tactile_th_batch, inputs_tactile_ff_batch, inputs_tactile_mf_batch,
             inputs_chamber_th_batch, inputs_chamber_ff_batch, inputs_chamber_mf_batch,
             inputs_press_batch, outputs_th_batch, outputs_ff_batch, outputs_mf_batch) in train_loader:
            optimizer.zero_grad()
            (inputs_tactile_th_batch, inputs_tactile_ff_batch, inputs_tactile_mf_batch,
             inputs_chamber_th_batch, inputs_chamber_ff_batch, inputs_chamber_mf_batch,
             inputs_press_batch, outputs_th_batch, outputs_ff_batch, outputs_mf_batch) = \
            (inputs_tactile_th_batch.to(device), inputs_tactile_ff_batch.to(device), inputs_tactile_mf_batch.to(device),
             inputs_chamber_th_batch.to(device), inputs_chamber_ff_batch.to(device), inputs_chamber_mf_batch.to(device),
             inputs_press_batch.to(device),
             outputs_th_batch.to(device), outputs_ff_batch.to(device), outputs_mf_batch.to(device))

            predictions_th, predictions_ff, predictions_mf =(
                model(inputs_tactile_th_batch, inputs_tactile_ff_batch, inputs_tactile_mf_batch,
                      inputs_chamber_th_batch, inputs_chamber_ff_batch, inputs_chamber_mf_batch,
                      inputs_press_batch))
            train_loss = (criterion(predictions_th, outputs_th_batch) +
                          criterion(predictions_ff, outputs_ff_batch) +
                          criterion(predictions_mf, outputs_mf_batch))

            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        if (epoch + 1) % args.save_interval == 0:
            save_path = args.save_dir + f'model_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} to {save_path}")

        average_distance_error_th_batch_list, average_distance_error_ff_batch_list, average_distance_error_mf_batch_list = [], [], []
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for (inputs_tactile_th_batch, inputs_tactile_ff_batch, inputs_tactile_mf_batch,
                 inputs_chamber_th_batch, inputs_chamber_ff_batch, inputs_chamber_mf_batch,
                 inputs_press_batch, outputs_th_batch, outputs_ff_batch, outputs_mf_batch) in val_loader:
                (inputs_tactile_th_batch, inputs_tactile_ff_batch, inputs_tactile_mf_batch,
                 inputs_chamber_th_batch, inputs_chamber_ff_batch, inputs_chamber_mf_batch,
                 inputs_press_batch, outputs_th_batch, outputs_ff_batch, outputs_mf_batch) = \
                (inputs_tactile_th_batch.to(device), inputs_tactile_ff_batch.to(device),
                 inputs_tactile_mf_batch.to(device),
                 inputs_chamber_th_batch.to(device), inputs_chamber_ff_batch.to(device),
                 inputs_chamber_mf_batch.to(device),
                 inputs_press_batch.to(device),
                 outputs_th_batch.to(device), outputs_ff_batch.to(device), outputs_mf_batch.to(device))
                predictions_th, predictions_ff, predictions_mf = (
                    model(inputs_tactile_th_batch, inputs_tactile_ff_batch, inputs_tactile_mf_batch,
                          inputs_chamber_th_batch, inputs_chamber_ff_batch, inputs_chamber_mf_batch,
                          inputs_press_batch))
                val_loss = (criterion(predictions_th, outputs_th_batch) +
                              criterion(predictions_ff, outputs_ff_batch) +
                              criterion(predictions_mf, outputs_mf_batch))
                running_val_loss += val_loss.item()

                error_th = np.abs((predictions_th.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min) -
                                  (outputs_th_batch.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min))
                error_ff = np.abs((predictions_ff.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min) -
                                  (outputs_ff_batch.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min))
                error_mf = np.abs((predictions_mf.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min) -
                                  (outputs_mf_batch.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min))
                distance_error_th, distance_error_ff, distance_error_mf = [], [], []
                for joint in range(5):
                    distance_error_th.append(math.sqrt(math.pow(error_th[0, joint, 0], 2) +
                                                       math.pow(error_th[0, joint, 1], 2) +
                                                       math.pow(error_th[0, joint, 2], 2)))
                    distance_error_ff.append(math.sqrt(math.pow(error_ff[0, joint, 0], 2) +
                                                       math.pow(error_ff[0, joint, 1], 2) +
                                                       math.pow(error_ff[0, joint, 2], 2)))
                    distance_error_mf.append(math.sqrt(math.pow(error_mf[0, joint, 0], 2) +
                                                       math.pow(error_mf[0, joint, 1], 2) +
                                                       math.pow(error_mf[0, joint, 2], 2)))
                average_distance_error_th_batch_list.append(np.mean(distance_error_th))
                average_distance_error_ff_batch_list.append(np.mean(distance_error_ff))
                average_distance_error_mf_batch_list.append(np.mean(distance_error_mf))

        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        average_distance_error_th = np.mean(average_distance_error_th_batch_list)
        average_distance_error_ff = np.mean(average_distance_error_ff_batch_list)
        average_distance_error_mf = np.mean(average_distance_error_mf_batch_list)
        average_distance_error_th_epoch_list.append(average_distance_error_th)
        average_distance_error_ff_epoch_list.append(average_distance_error_ff)
        average_distance_error_mf_epoch_list.append(average_distance_error_mf)

        print(f"Epoch {epoch+1}/{args.epochs},"
              f"Training Loss: {avg_train_loss},"
              f"Validation Loss: {avg_val_loss},")
        print(f"Average Distance Error for Thumb: {average_distance_error_th},"
              f"Average Distance Error for First Finger: {average_distance_error_ff},"
              f"Average Distance Error for Middle Finger: {average_distance_error_mf}")

    plt.figure(figsize=(7,5), dpi=600)
    axis_font = {'weight':'bold','size':14}
    title_font = {'weight':'bold','size':15}
    plt.rcParams.update({'font.size':13})
    plt.rcParams["font.weight"] ="bold"

    plt.plot(average_distance_error_th_epoch_list, label='Error for Thumb')
    plt.plot(average_distance_error_ff_epoch_list, label='Error for First Finger')
    plt.plot(average_distance_error_mf_epoch_list, label='Error for Middle Finger')
    plt.title(label='Distance Error Over Epochs', fontdict=title_font)
    plt.xlabel(xlabel='Epochs', fontdict=axis_font)
    plt.ylabel(ylabel='Error (m)', fontdict=axis_font)
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.show()

    plt.plot(train_loss_history, label='Training Loss', color='#b01f24')
    plt.plot(val_loss_history, label='Validation Loss', color='#003f88')
    plt.title(label='Training and Validation Loss Over Epochs', fontdict=title_font)
    plt.xlabel(xlabel='Epochs', fontdict=axis_font)
    plt.ylabel(ylabel='Loss', fontdict=axis_font)
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.show()

    model.eval()
    test_loss = 0.0
    average_distance_error_th_test_list, average_distance_error_ff_test_list, average_distance_error_mf_test_list = [], [], []
    average_distance_error_tracker1_list, average_distance_error_tracker2_list, average_distance_error_tracker3_list, \
    average_distance_error_tracker4_list, average_distance_error_tracker5_list = [], [], [], [], []
    with ((torch.no_grad())):
        for (inputs_tactile_th_batch, inputs_tactile_ff_batch, inputs_tactile_mf_batch,
                 inputs_chamber_th_batch, inputs_chamber_ff_batch, inputs_chamber_mf_batch,
                 inputs_press_batch, outputs_th_batch, outputs_ff_batch, outputs_mf_batch) in test_loader:
             (inputs_tactile_th_batch, inputs_tactile_ff_batch, inputs_tactile_mf_batch,
              inputs_chamber_th_batch, inputs_chamber_ff_batch, inputs_chamber_mf_batch,
              inputs_press_batch, outputs_th_batch, outputs_ff_batch, outputs_mf_batch) = \
             (inputs_tactile_th_batch.to(device), inputs_tactile_ff_batch.to(device),
              inputs_tactile_mf_batch.to(device),
              inputs_chamber_th_batch.to(device), inputs_chamber_ff_batch.to(device),
              inputs_chamber_mf_batch.to(device),
              inputs_press_batch.to(device),
              outputs_th_batch.to(device), outputs_ff_batch.to(device), outputs_mf_batch.to(device))
             predictions_th, predictions_ff, predictions_mf = (
                 model(inputs_tactile_th_batch, inputs_tactile_ff_batch, inputs_tactile_mf_batch,
                       inputs_chamber_th_batch, inputs_chamber_ff_batch, inputs_chamber_mf_batch,
                       inputs_press_batch))
             test_loss += (criterion(predictions_th, outputs_th_batch).item() +
                           criterion(predictions_ff, outputs_ff_batch).item() +
                           criterion(predictions_mf, outputs_mf_batch).item())
             error_th = np.abs(
                 (predictions_th.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min) -
                 (outputs_th_batch.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min))
             error_ff = np.abs(
                 (predictions_ff.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min) -
                 (outputs_ff_batch.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min))
             error_mf = np.abs(
                 (predictions_mf.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min) -
                 (outputs_mf_batch.detach().cpu().numpy() * (args.norm_max - args.norm_min) + args.norm_min))
             distance_error_th, distance_error_ff, distance_error_mf = [], [], []
             for joint in range(5):
                 distance_error_th.append(math.sqrt(math.pow(error_th[0, joint, 0], 2) +
                                                   math.pow(error_th[0, joint, 1], 2) +
                                                   math.pow(error_th[0, joint, 2], 2)))
                 distance_error_ff.append(math.sqrt(math.pow(error_ff[0, joint, 0], 2) +
                                                   math.pow(error_ff[0, joint, 1], 2) +
                                                   math.pow(error_ff[0, joint, 2], 2)))
                 distance_error_mf.append(math.sqrt(math.pow(error_mf[0, joint, 0], 2) +
                                                   math.pow(error_mf[0, joint, 1], 2) +
                                                   math.pow(error_mf[0, joint, 2], 2)))
             distance_error_joint1 = math.sqrt(math.pow(error_th[0, 0, 0], 2) + math.pow(error_th[0, 0, 1], 2) + math.pow(error_th[0, 0, 2], 2)) + \
                                     math.sqrt(math.pow(error_ff[0, 0, 0], 2) + math.pow(error_ff[0, 0, 1], 2) + math.pow(error_ff[0, 0, 2], 2)) + \
                                     math.sqrt(math.pow(error_mf[0, 0, 0], 2) + math.pow(error_mf[0, 0, 1], 2) + math.pow(error_mf[0, 0, 2], 2))
             distance_error_joint2 = math.sqrt(math.pow(error_th[0, 1, 0], 2) + math.pow(error_th[0, 1, 1], 2) + math.pow(error_th[0, 1, 2], 2)) + \
                                     math.sqrt(math.pow(error_ff[0, 1, 0], 2) + math.pow(error_ff[0, 1, 1], 2) + math.pow(error_ff[0, 1, 2], 2)) + \
                                     math.sqrt(math.pow(error_mf[0, 1, 0], 2) + math.pow(error_mf[0, 1, 1], 2) + math.pow(error_mf[0, 1, 2], 2))
             distance_error_joint3 = math.sqrt(math.pow(error_th[0, 2, 0], 2) + math.pow(error_th[0, 2, 1], 2) + math.pow(error_th[0, 2, 2], 2)) + \
                                     math.sqrt(math.pow(error_ff[0, 2, 0], 2) + math.pow(error_ff[0, 2, 1], 2) + math.pow(error_ff[0, 2, 2], 2)) + \
                                     math.sqrt(math.pow(error_mf[0, 2, 0], 2) + math.pow(error_mf[0, 2, 1], 2) + math.pow(error_mf[0, 2, 2], 2))
             distance_error_joint4 = math.sqrt(math.pow(error_th[0, 3, 0], 2) + math.pow(error_th[0, 3, 1], 2) + math.pow(error_th[0, 3, 2], 2)) + \
                                     math.sqrt(math.pow(error_ff[0, 3, 0], 2) + math.pow(error_ff[0, 3, 1], 2) + math.pow(error_ff[0, 3, 2], 2)) + \
                                     math.sqrt(math.pow(error_mf[0, 3, 0], 2) + math.pow(error_mf[0, 3, 1], 2) + math.pow(error_mf[0, 3, 2], 2))
             distance_error_joint5 = math.sqrt(math.pow(error_th[0, 4, 0], 2) + math.pow(error_th[0, 4, 1], 2) + math.pow(error_th[0, 4, 2], 2)) + \
                                     math.sqrt(math.pow(error_ff[0, 4, 0], 2) + math.pow(error_ff[0, 4, 1], 2) + math.pow(error_ff[0, 4, 2], 2)) + \
                                     math.sqrt(math.pow(error_mf[0, 4, 0], 2) + math.pow(error_mf[0, 4, 1], 2) + math.pow(error_mf[0, 4, 2], 2))
             average_distance_error_th_test_list.append(np.mean(distance_error_th))
             average_distance_error_ff_test_list.append(np.mean(distance_error_ff))
             average_distance_error_mf_test_list.append(np.mean(distance_error_mf))
             average_distance_error_tracker1_list.append(distance_error_joint1)
             average_distance_error_tracker2_list.append(distance_error_joint2)
             average_distance_error_tracker3_list.append(distance_error_joint3)
             average_distance_error_tracker4_list.append(distance_error_joint4)
             average_distance_error_tracker5_list.append(distance_error_joint5)

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")
    average_distance_test_error_th = np.mean(average_distance_error_th_test_list)
    average_distance_test_error_ff = np.mean(average_distance_error_ff_test_list)
    average_distance_test_error_mf = np.mean(average_distance_error_mf_test_list)
    average_distance_error_tracker1 = np.mean(average_distance_error_tracker1_list)
    average_distance_error_tracker2 = np.mean(average_distance_error_tracker2_list)
    average_distance_error_tracker3 = np.mean(average_distance_error_tracker3_list)
    average_distance_error_tracker4 = np.mean(average_distance_error_tracker4_list)
    average_distance_error_tracker5 = np.mean(average_distance_error_tracker5_list)
    print(f"Average Distance Test Error for Thumb: {average_distance_test_error_th},"
          f"Average Distance Test Error for First Finger: {average_distance_test_error_ff},"
          f"Average Distance Test Error for Middle Finger: {average_distance_test_error_mf}")
    print(f"Average Distance Test Error: {(average_distance_test_error_th + average_distance_test_error_ff + average_distance_test_error_mf)/3}")
    print(f"Average Distance Test Error for Tracker1: {average_distance_error_tracker1/3},"
          f"Average Distance Test Error for Tracker2: {average_distance_error_tracker2/3},"
          f"Average Distance Test Error for Tracker3: {average_distance_error_tracker3/3},"
          f"Average Distance Test Error for Tracker4: {average_distance_error_tracker4/3},"
          f"Average Distance Test Error for Tracker5: {average_distance_error_tracker5/3},")


if __name__ == '__main__':
    main()
