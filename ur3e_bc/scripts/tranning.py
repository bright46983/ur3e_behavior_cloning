import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ur3e_bc.modules import UR3EDataset
from ur3e_bc.models import UR3EBCModel
from torch.utils.data import DataLoader

import time


def train_model(model, dataset, num_epochs=20, batch_size=16, lr=1e-4,
                checkpoint_path='checkpoint.pth', resume_training=False,save_path='model.pth',
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    # Split dataset
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.to(device)

    vel_criterion = nn.MSELoss()
    pose_criterion = nn.MSELoss()
    state_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TensorBoard Writer
    log_dir = f"runs/ur3e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    start_epoch = 0
    best_val_vel = float('inf')

    # Resume from checkpoint
    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_vel = checkpoint.get('best_val_vel', float('inf'))
        print(f"✅ Loaded checkpoint from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"🚀 Epoch {epoch+1}/{num_epochs}"):
            front = batch['front_cam'].to(device)
            side = batch['side_cam'].to(device)
            hand = batch['hand_cam'].to(device)
            ee_pose = batch['ee_pose'].to(device)
            joints = batch['joints'].to(device)

            vel_gt = batch['ee_velocity'].to(device)
            pose_gt = batch['hole_pose'].to(device)
            state_gt = batch['state'].squeeze(-1).to(device)
            optimizer.zero_grad()

            vel_pred, pose_pred, state_pred = model(front, side, hand, ee_pose, joints)

            loss_vel = vel_criterion(vel_pred, vel_gt)
            loss_pose = pose_criterion(pose_pred, pose_gt)
            loss_state = state_criterion(state_pred, state_gt)

            total_loss = loss_vel * 1.0 + loss_pose * 0.5 + loss_state * 0.2
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"📊 Train Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        # Validation
        val_vel, val_pose, val_state, val_total, val_acc = evaluate_model(model, val_loader, device)
        print(f"✅ Validation — Vel MSE: {val_vel:.4f} | Pose MSE: {val_pose:.4f} | State Acc: {val_acc*100:.2f}%")

        # Train logs
        writer.add_scalar("Train/Velocity_Loss", loss_vel.item(), epoch)
        writer.add_scalar("Train/Pose_Loss", loss_pose.item(), epoch)
        writer.add_scalar("Train/State_Loss", loss_state.item(), epoch)
        writer.add_scalar("Train/Total_Loss", total_loss.item(), epoch)

        # Validation logs
        writer.add_scalar("Val/Velocity_Loss", val_vel, epoch)
        writer.add_scalar("Val/Pose_Loss", val_pose, epoch)
        writer.add_scalar("Val/State_Loss", val_state, epoch)
        writer.add_scalar("Val/Total_Loss", val_total, epoch)
        writer.add_scalar("Val/State_Accuracy", val_acc, epoch)

        # Save best model
        # if val_vel < best_val_vel:
        #     best_val_vel = val_vel
        #     torch.save(model.state_dict(), "best_model.pth")
        #     print("🏆 Best model updated and saved (based on velocity loss)")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_vel': best_val_vel
        }, save_path)
        print(f"💾 Checkpoint saved at epoch {epoch+1}")

    writer.close()


@torch.no_grad()
def evaluate_model(model, val_loader, device):
    model.eval()
    vel_criterion = nn.MSELoss()
    pose_criterion = nn.MSELoss()
    state_criterion = nn.CrossEntropyLoss()

    total_loss, total_vel_loss, total_pose_loss, total_state_loss = 0, 0, 0, 0
    correct, total = 0, 0

    for batch in val_loader:
        front = batch['front_cam'].to(device)
        side = batch['side_cam'].to(device)
        hand = batch['hand_cam'].to(device)
        ee_pose = batch['ee_pose'].to(device)
        joints = batch['joints'].to(device)

        vel_gt = batch['ee_velocity'].to(device)
        pose_gt = batch['hole_pose'].to(device)
        state_gt = batch['state'].squeeze(-1).to(device)

        vel_pred, pose_pred, state_pred = model(front, side, hand, ee_pose, joints)

        loss_vel = vel_criterion(vel_pred, vel_gt)
        loss_pose = pose_criterion(pose_pred, pose_gt)
        loss_state = state_criterion(state_pred, state_gt)
    
        total_vel_loss += loss_vel.item()
        total_pose_loss += loss_pose.item()
        total_state_loss += loss_state.item()

        # Compute classification accuracy
        preds = torch.argmax(state_pred, dim=1)
        correct += (preds == state_gt).sum().item()
        total += state_gt.size(0)

    val_vel_loss = total_vel_loss / len(val_loader)
    val_pose_loss = total_pose_loss / len(val_loader)
    val_state_loss = total_state_loss / len(val_loader)
    total_val_loss = val_vel_loss * 1.0 + val_pose_loss * 0.5 + val_state_loss * 0.2

    accuracy = correct / total

    return val_vel_loss, val_pose_loss, val_state_loss, total_val_loss, accuracy


class Args:
    data_dir = "/home/tanakrit-ubuntu/ur3e_mujoco_tasks/scripts/data"
    num_epochs = 100
    batch_size = 16
    lr = 1e-4
    save_dir = "/home/tanakrit-ubuntu/ur3e_behavior_cloning/runs"
    model_name = "model"
    checkpoint_path = ""
    resume_training = False

if __name__ == '__main__':
    # load arguments 
    args = Args()

    # load dataset 
    dataset = UR3EDataset(args.data_dir)

    # Initiate model
    model = UR3EBCModel() 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"{args.model_name}_{timestamp}.pth")
    

    train_model(model, dataset, num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr,
                checkpoint_path=args.checkpoint_path, resume_training=args.resume_training, save_path=save_path,
                device='cuda' if torch.cuda.is_available() else 'cpu')
    
