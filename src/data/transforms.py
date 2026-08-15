from torchvision import transforms


def build_transforms(image_size: int):
    default_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_ae = transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ColorJitter(saturation=0.2)
    ])

    def train_transform(image):
        return default_transform(image), default_transform(transform_ae(image))
    
    return default_transform, train_transform


def build_penalty_transforms(image_size: int):
    penalty_transform = transforms.Compose([
        transforms.Resize((2 * image_size, 2 * image_size)),
        transforms.RandomGrayscale(0.3),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    return penalty_transform