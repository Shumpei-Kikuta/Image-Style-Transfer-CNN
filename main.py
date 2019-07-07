import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

CONTENT_PATH = "City.jpg"
STYLE_PATH = "StarryNight.jpg"

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
    image = image.transpose(1, 2, 0) #tensorの軸の順番を入れ替える
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) #denormalize
    image = image.clip(0, 1) #0-1の間におさめる
    return image

def get_features(image, model):
    """inputをCNNに入れた時の途中の値を辞書にして返す"""
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
    """グラム行列によってstyleの類似度を計算する"""
    _, d, h, w = tensor.size() #batch, depth, height, weight
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def calc_style_loss(style_weights, target_features, style_grams):
    """styleのloss"""
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        _, depth, height, width = target_feature.shape
        style_loss += layer_style_loss / (depth * height * width)
    return style_loss

def main(device):
    # 学習済みモデル
    vgg = models.vgg19(pretrained=True).features # CNNの部分だけ
    for param in vgg.parameters():
      param.requires_grad_(False) #パラメータは固定
    vgg.to(device)

    # contentとstyleの画像
    content = load_image(CONTENT_PATH).to(device)
    style = load_image(STYLE_PATH, shape=content.shape[-2:]).to(device)

    # 特徴量の抽出
    content_features = get_features(content, vgg) #{"conv1_1": [[~]]}
    style_features = get_features(style, vgg)

    # styleのグラムマトリックス
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    content_weight = 1
    style_weight = 1e6
    steps = 9000
    show_every = 300
    style_weights = {
        "conv1_1": 1, 
        "conv2_1": .75, 
        "conv3_1": 0.2,
        "conv4_1": 0.2, 
        "conv5_1": 0.2
    }

    #初期値はcontentの画像
    target = content.clone().requires_grad_(True).to(device) 

    # targetを更新して行く
    optimizer = optim.Adam([target], lr=0.003)

    for ii in range(1, steps+1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]))**2
        style_loss = calc_style_loss(style_weights, target_features, style_grams)

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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    main(device)