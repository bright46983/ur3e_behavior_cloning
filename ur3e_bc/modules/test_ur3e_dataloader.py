import cv2
import numpy as np
from ur3e_dataloader import UR3EDataset
from torch.utils.data import Dataset, DataLoader


def visualize_batch(batch, batch_idx=0):
    """
    Visualizes a batch of data from the dataloader in a single window.

    Args:
        batch (dict): A batch of data from the dataloader.
        batch_idx (int): Index within the batch to visualize (default: 0).
    """
    # Extract batch data for the selected sample
    front_cam = batch["front_cam"][batch_idx].cpu().numpy()  # (3, H, W, 3)
    side_cam = batch["side_cam"][batch_idx].cpu().numpy()  # (3, H, W, 3)
    hand_cam = batch["hand_cam"][batch_idx].cpu().numpy()  # (3, H, W, 3)

    # Convert images to OpenCV format (H, W, C) and scale up for visibility
    def preprocess_img(img):
        """
        Preprocess image from [f, H, W, C] to OpenCV format [H, W, C] for display.
        """
        # img = np.transpose(img, (1, 2, 0))  # Convert [f, H, W, C] -> [H, W, f, C] -> [H, W, C]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Convert back to 0-255 for display
        return img  # Return as H, W, C

    # Process each camera's image and time step (f=3)
    front_cam_imgs = [preprocess_img(front_cam[i]) for i in range(3)]  # (t-2, t-1, t)
    side_cam_imgs = [preprocess_img(side_cam[i]) for i in range(3)]
    hand_cam_imgs = [preprocess_img(hand_cam[i]) for i in range(3)]

    # Stack images in one grid (3x3)
    # The layout of the images will be:
    # front#t-2, front#t-1, front#t
    # side#t-2, side#t-1, side#t
    # hand#t-2, hand#t-1, hand#t

    top_row = np.concatenate(front_cam_imgs, axis=1)  # Concatenate the 3 front camera frames
    middle_row = np.concatenate(side_cam_imgs, axis=1)  # Concatenate the 3 side camera frames
    bottom_row = np.concatenate(hand_cam_imgs, axis=1)  # Concatenate the 3 hand camera frames

    # Combine all rows vertically
    all_images = np.vstack([top_row, middle_row, bottom_row])

    # Display the images in one window
    cv2.imshow("Camera Views: Front, Side, Hand (t-2, t-1, t)", all_images)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()  # Close the window after key press

    # Optionally print numerical data for the current sample
    ee_pose = batch["ee_pose"][batch_idx].cpu().numpy()  # (3,7)
    ee_velocity = batch["ee_velocity"][batch_idx].cpu().numpy()  # (6)
    hole_pose = batch["hole_pose"][batch_idx].cpu().numpy()  # (7)
    state = batch["state"][batch_idx].cpu().numpy()  # (1)

    print("\nEnd Effector Pose Over Time:")
    print(f"  t-2: {ee_pose[0]}")
    print(f"  t-1: {ee_pose[1]}")
    print(f"  t:   {ee_pose[2]}")
    
    print("\nEnd Effector Velocity:")
    print(f"  [vx, vy, vz, wx, wy, wz]: {ee_velocity}")
    
    print("\nHole Position and Orientation:")
    print(f"  [x, y, z, qx, qy, qz, qw]: {hole_pose}")

    # Print the current state
    state_labels = {0: "Move", 1: "Insert", 2: "Rotate"}
    print(f"\nCurrent State: {state_labels.get(state.item(), 'Unknown')} ({state.item()})")


# Example usage
zip_dir = "/home/tanakrit-ubuntu/ur3e_mujoco_tasks/scripts/data"
batch_size = 16

dataset = UR3EDataset(zip_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Example usage with a batch from the dataloader
for batch in dataloader:
    print(batch["front_cam"].shape)
    visualize_batch(batch, batch_idx=0)
