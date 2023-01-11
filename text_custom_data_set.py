import  torch
import spacy
from torchvision.transforms import  transforms
from torch.utils.data import  DataLoader, Dataset
import pandas as pd
from PIL import  Image
from skimage import io
from torch.nn.utils.rnn import pad_sequence
# from spacy.cli.download import download
# download(model="en_core_web_sm")
spacy_en=spacy.load("en_core_web_sm")
class Vocabulary():
    def __init__(self,sentences,threshold=5):
        self.sentences=sentences
        self.threshold=threshold
        self.stoi={'<PAD>':0,'<SOS>':1,'<UKN>':2,'<EOS>':3}
        self.itos={0:'<PAD>',1:'<SOS>',2:'<UKN>',3:'<EOS>'}
    def buil_vocabulary(self):
        indx=4
        frequencies={}
        for sentence in self.sentences:
            # print(sentence)
            for word in self.tokenized(sentence):
                if word not in frequencies:
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                if frequencies[word]==self.threshold:
                    self.stoi[word]=indx
                    self.itos[indx]=word
                    indx += 1
        # print('Finished!')
    def numricalized(self,sentence):
        return [self.stoi[word] if word in self.stoi else self.stoi['<UKN>']
                for word in sentence]

    @staticmethod
    def tokenized(text_):
        # for tok in spacy_en.tokenizer(text_):
        #     print(tok.text.lower())
        return [tok.text.lower() for tok in spacy_en.tokenizer(text_)]

import  os
class FlickerDataset(Dataset):
    def __init__(self,root_dir, caption_file, transform=None):
        self.root_dir=root_dir
        self.df=pd.read_csv(caption_file)
        self.transform=transform
        self.img_id=self.df['image']
        self.caption_id=self.df['caption']
        self.vocab = Vocabulary(self.caption_id.tolist())
        self.vocab.buil_vocabulary()

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        thershold=5
        imgs_=self.img_id[index]
        img=Image.open(os.path.join(self.root_dir,imgs_)).convert("RGB")
        if self.transform is not None:
            img=self.transform(img)

        caption=self.caption_id[index]
        caption_=[]
        caption_+=self.vocab.tokenized(caption)
        caption_.append('<EOS>')
        numericalized_=self.vocab.numricalized(caption_)
        return img,torch.tensor(numericalized_)

class My_Collate():
    def __init__(self,pad_inx):
        self.pad_inx=pad_inx
    def __call__(self, batch):
        img=[item[0].unsqueeze(0) for item in batch]
        img=torch.cat(img,dim=0)
        target=[item[1] for item in batch]
        target=pad_sequence(target,batch_first=False,padding_value=self.pad_inx)
        return img,torch.tensor(target)

def get_loader(root_dir,caption_file,transform_=None,batch=32,shuffle=True,num_worker=0):

    dataset=FlickerDataset(root_dir=root_dir,caption_file=caption_file,transform=transform_)
    pad_indx =dataset.vocab.stoi['<PAD>']
    load=DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=0,collate_fn=My_Collate(pad_inx=pad_indx))
    return load,dataset
if __name__=='__main__':
    root_dir='flickr8k/images'; caption_file='flickr8k/captions.txt'
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    load,_=get_loader(root_dir,caption_file,transform_=transform)
    for _,(img,caption_value) in enumerate(load):
        print(caption_value)




