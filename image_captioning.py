import torch
import torch.nn as nn
import text_custom_data_set as txt_caption
from torchvision import  models
class Encoder (nn.Module):
    def __init__(self,embed_size,train_c=False):
        super(Encoder, self).__init__()
        self.train_c=train_c
        self.relu=nn.ReLU()
        self.droup_out=nn.Dropout(p=0.5)
        self.inception=models.inception_v3(pretrained=True,aux_logits=True)
        self.inception.fc=nn.Linear(self.inception.fc.in_features,embed_size)
        # print(self.model)
    def forward(self,img):
        features=self.inception(img)
        # print(features[0].shape)
        for name, param in self.inception.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                param.requires_grad=True
            else:
                param.requires_grad=self.train_c
        return self.droup_out(self.relu(features[0]))
class Decoder(nn.Module):
    def __init__(self,embed_size,hidden_size, num_layers, vocab_size):
        super(Decoder, self).__init__()
        self.embeding=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers)
        self.output_ = nn.Linear(hidden_size, vocab_size)
        self.drop_out=nn.Dropout(p=0.5)
    def forward(self, features,captions):
        embeding_=self.drop_out(self.embeding(captions))
        embed=torch.cat((features.unsqueeze(0),embeding_),dim=0)
        hidden,_=self.lstm(embeding_)
        return self.output_(hidden)
class CNN_to_RNN(nn.Module):
    def __init__(self,embed_size,hidden_size, num_layers, vocab_size):
        super(CNN_to_RNN, self).__init__()
        self.encoder=Encoder(embed_size)
        self.decoder=Decoder(embed_size,hidden_size, num_layers, vocab_size)
        self.droup_out=nn.Dropout(p=0.5)
    def forward(self,img, caption):
        features = self.encoder(img)
        embeding_ = self.decoder(features,caption)
        return embeding_
    def caption_image(self,img,vocabulary,max_length=50):
        results_ = []
        with torch.no_grad:
            feature = self.encoder(img)
            state = None
            hidden, state = self.decoder.lstm(feature, state)
            for _index in range(max_length):
                prediction = self.decoder.output_(hidden)
                predict = prediction.argmax(1)
                if vocabulary.itos[predict.item()] == '<EOS>':
                    break
                else:
                    results_.append(vocabulary.itos[predict.item()])
                hidden= self.decoder.embeding(predict)
                state = hidden
            return [vocabulary.itos[item] for item in results_]













if __name__=='__main__':
    x=torch.rand((3,3,299,299))
    models=Encoder(256)
    r=models(x)
    print(r)

