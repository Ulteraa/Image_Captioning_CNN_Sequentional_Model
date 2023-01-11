import torch
import torch.optim as optim
import torchvision.transforms as transform
import text_custom_data_set as CT
import image_captioning as IC
import torch.nn as nn
def train():
    transform_=transform.Compose([transform.Resize((356,356)),
    transform.RandomCrop((299,299)), transform.ToTensor(),
    transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    batch = 32
    root_dir='flickr8k/images';caption_file='flickr8k/captions.txt'
    data_loader,dataset=CT.get_loader(root_dir, caption_file, transform_=transform_, batch=batch, shuffle=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_size=256
    print(dataset.vocab)
    vocab_size=len(dataset.vocab.stoi)
    num_layer=1
    hidden_size=256
    epochs=100
    lr=0.001
    load_model=False
    save_model=True
    model=IC.CNN_to_RNN(embed_size,hidden_size, num_layer, vocab_size)
    criterion=nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    opimizer=optim.Adam(model.parameters(),lr=lr)
    model.train()
    if load_model:
        pass
    for epoch in range(epochs):
        for _ , (img,caption) in enumerate(data_loader):
            img=img.to(device)
            caption=caption.to(device)
            output_=model(img,caption)
            loss=criterion(output_.reshape(-1, output_.shape[2]),caption.reshape(-1))
            opimizer.zero_grad()
            loss.backward(loss)
            opimizer.step()
            print(f'the loss is: {loss}')
if __name__=='__main__':
    train()




