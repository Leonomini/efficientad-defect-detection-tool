import torch
from tqdm import tqdm
from src.inference.inference import compute_anomaly_maps


# Method computes teacher channel normalization parameters (mean μ and standard deviation σ)
# For computation only the first (non-augmented) image is used.
@torch.no_grad()
def teacher_normalization(
    teacher, 
    train_loader,
    device,
):
    mean_outputs = []

    for train_image, _ in tqdm(train_loader, desc="Computing mean of features"):
        
        train_image = train_image.to(device)
        
        teacher_output = teacher(train_image)
        mean_outputs.append(torch.mean(teacher_output, dim=[0, 2, 3]))

    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)[None, :, None, None]

    mean_distances = []

    for train_image, _ in tqdm(train_loader, desc="Computing std of features"):

        train_image = train_image.to(device)
        
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distances.append(torch.mean(distance, dim=[0, 2, 3]))

    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_std = torch.sqrt(channel_var[None, :, None, None])

    return channel_mean, channel_std


# Method computes quantile-based map normalization on validation images.
# For computation only the first (non-augmented) image is used.
@torch.no_grad()
def map_normalization(
    validation_loader, 
    teacher, 
    student, 
    autoencoder,
    teacher_mean, 
    teacher_std,
    device,
    q_start,
    q_end,
    leave_pbar: bool = False,
):
    maps_st = []
    maps_ae = []

    for image, _ in tqdm(validation_loader, desc="Map normalization", leave=leave_pbar):
        image = image.to(device)

        _, map_st, map_ae = compute_anomaly_maps(
            image=image, 
            teacher=teacher, 
            student=student,
            autoencoder=autoencoder, 
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
        )

        maps_st.append(map_st)
        maps_ae.append(map_ae)
        
    maps_st = torch.cat(maps_st).flatten()
    maps_ae = torch.cat(maps_ae).flatten()

    qa_st = torch.quantile(maps_st, q=q_start)
    qb_st = torch.quantile(maps_st, q=q_end)
    qa_ae = torch.quantile(maps_ae, q=q_start)
    qb_ae = torch.quantile(maps_ae, q=q_end)

    return qa_st, qb_st, qa_ae, qb_ae
