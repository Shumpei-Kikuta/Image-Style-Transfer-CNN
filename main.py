import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_image(img_path, max_size=400, shape=None):
    """画像を読込む"""
    image = Image.open(img_path).convert("RGB")
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = in_transform(image).unsqueeze(0)
    return image

def im_convert(tensor):
    """convert tensor to image"""
    image = tensor.cpu().clone().detach().numpy()
    image = image.squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

def get_features(image, model):
    layers = {
        "0": "conv1_1", 
        "5": "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1", 
        "21": "conv4_2", #content extraction
        "28": "conv5_1" 
            }
            
    features = {}
    for name, layer in model._modules.items():
        image = layer(image)
    if name in layers:
        features[layers[name]] = image
    return features

def gram_matrix(tensor):
    b, d, h, w = tensor.size() #batch, depth, height, weight
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# 学習済みモデルのimport 
vgg = models.vgg19(pretrained=True).features # CNNの部分だけ

for param in vgg.parameters():
    param.require_grad_(False) #重みをいじらない

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
vgg.to(device)

# contentとstyleの画像
content = load_image("City.jpg").to(device)
style = load_image("StarryNight.jpg", shape=content.shape[-2:]).to(device)

# 画像の出力
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax1.axis=("off")
ax2.imshow(im_convert(style))
ax2.axis=("off")

# 特徴量の抽出
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

style_weights = {
    "conv1_1": 1, 
    "conv2_1": .75, 
    "conv3_1": 0.2,
    "conv4_1": 0.2, 
    "conv5_1": 0.2
}

content_weight = 1
style_weight = 1e6

target  = content.clone().requires_grad_(True).to(device)


show_every = 300
optimizer = optim.Adam([target], lr=0.003)
steps = 2100

height, width, channels = im_convert(target).shape
image_array = np.empty(shape=(300, height, width, channels))
capture_frame = steps/300
counter = 0

for ii in range(1, steps+1):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]))**2
    style_loss = 0

    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        _, depth, height, weight = target_feature.shape
        style_loss += layer_style_loss / (depth * height * width)


    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
  
    if ii % show_every == 0:
        print("Total loss: ".format(total_loss.item()))
        print("Iteration: {}".format(ii))

        plt.imshow(im_convert(target))
        plt.axis("off")
        plt.show()
    
    if ii % capture_frame == 0:
        image_array[counter] = im_convert(target)
        counter += 1
