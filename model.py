from transformers import ResNetConfig,ResNetModel
from torch import nn  
from torch.nn import TransformerDecoder,TransformerDecoderLayer
class Model(nn.modules):
     
    def __init__(self,in_ch,embed_size,hidden_size,depth,de_d_model,de_numhead,numlayer,type='full'):
        super(nn.modules, self).__init__()
        self.resconfig=ResNetConfig(num_channels=in_ch,embedding_size=embed_size,hidden_sizes=hidden_size,depths=depth)
        self.encoder=ResNetModel(self.resconfig)
        self.decoderlayer=TransformerDecoderLayer(d_model=de_d_model,nhead=de_numhead)
        self.decoder=TransformerDecoder(self.decoder,num_layers=numlayer)
        self.type=type
    def forward(self,input,gt):
        hidden=self.encoder(input)
        if self.type=="full":
            output=self.decoder(hidden.pooled_output,gt)
        elif  self.type=="patch":
            hidden=hidden.last_hidden_state.view(hidden.shape[0],hidden.shape[1],-1).permute(0,2,1)
            output=self.decoder(hidden,gt[:,:-1])
        return output
