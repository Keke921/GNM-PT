import timm

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed
from .vit import VisionTransformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

from torch.distributions import normal

import math
# class Prompt(nn.Module):
#     def __init__(self, type='Deep', depth=12, channel=768, length=10):
#         super().__init__()
#         self.prompt = nn.Parameter(torch.zeros(depth, length, channel))
#         trunc_normal_


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        #self.bias = torch.nn.Parameter(torch.empty(out_features)) #加了没啥用
        #fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        #bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        #cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        cosine = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1)) #, self.bias
        return cosine

class PromptLearner(nn.Module):
    def __init__(self, config, num_classes, prompt_length, prompt_depth, prompt_channels, normed=True, merge_prompt=False):
        super().__init__()
        self.Prompt_Tokens = nn.Parameter(torch.zeros(prompt_depth, prompt_length, prompt_channels))
        if normed == True:
            self.head = NormedLinear(prompt_channels, num_classes)  
        else:
            self.head = nn.Linear(prompt_channels, num_classes)           
        trunc_normal_(self.head.weight, std=.02)
        trunc_normal_(self.Prompt_Tokens, std=.02)
        
        if merge_prompt:
            self.tokens_proj = nn.Linear((prompt_length+1)*prompt_channels, prompt_channels)
            trunc_normal_(self.tokens_proj.weight, std=.02)
        
    def forward(self, x):
        return self.head(x)

class VPT_ViT(VisionTransformer):
    def __init__(self, config, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', Prompt_Token_num=1, VPT_type="Deep"):

        # Recreate ViT
        self.config = config
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                         representation_size, distilled, drop_rate, attn_drop_rate, drop_path_rate, embed_layer,
                         norm_layer, act_layer, weight_init)
        self.VPT_type = VPT_type
        # if VPT_type == "Deep":
        #     self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        # else:  # "Shallow"
        #     self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))
        # trunc_normal_(self.Prompt_Tokens, std=.02)
        self.dropout = nn.Dropout(0.1)
        self.prompt_learner = PromptLearner(self.config, self.config.num_classes, Prompt_Token_num, depth, embed_dim, 
                                            config.normed, config.merge_prompt)
        self.load_prompt() 
        if config.temperature:
            self.temp = nn.Parameter(torch.ones(12)) #temp_num=12
        
        
        self.config = config
    def Freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        # self.Prompt_Tokens.requires_grad_(True)
        for param in self.prompt_learner.parameters():
            param.requires_grad_(True)
        #self.temp.requires_grad_(True)


    def Freeze_Prompt(self):
        for param in self.parameters():
            param.requires_grad_(False)
        # self.Prompt_Tokens.requires_grad_(True)
        self.Prompt_Tokens.requires_grad_(False)
        self.head.requires_grad_(True)        


    def load_prompt(self): #, prompt_state_dict
        self.classifier = self.prompt_learner.head #prompt_state_dict['head']
        self.Prompt_Tokens = self.prompt_learner.Prompt_Tokens #prompt_state_dict['Prompt_Tokens']
        if self.config.merge_prompt:
            self.tokens_proj = self.prompt_learner.tokens_proj

    def reinit_temp(self):
        #assert self.temp_learn, "reinit_temp() could be run only when config.TEMP_LEARN == True"
        self.temp.data.copy_(self.temp.data.clamp(min=0.01, max=10.00))
        #self.prompt_config.TEMP_MIN=0.01  self.prompt_config.TEMP_MAX=10.0


    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":
            Prompt_Token_num = self.Prompt_Tokens.shape[1]
            #Prompt_out = torch.zeros([self.Prompt_Tokens.shape[0]+1, x.shape[0],self.Prompt_Tokens.shape[1],
            #                          self.Prompt_Tokens.shape[2]], device='cuda')
            #if self.config.merge_prompt:
            #    Prompt_out = torch.zeros(x.shape[0], 1, self.Prompt_Tokens.shape[2], device='cuda')
            #    Prompt_Token_num = Prompt_Token_num+1
            Prompt_out_end = torch.zeros(x.shape[0], self.Prompt_Tokens.shape[1], self.Prompt_Tokens.shape[2], device='cuda')
            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                temp = 1 #self.temp if not isinstance(self.temp, nn.Parameter) else self.temp[i]
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                Prompt_in = Prompt_Tokens.expand(x.shape[0], -1, -1)
                #if self.config.merge_prompt:    
                #    Prompt_in = torch.cat((Prompt_in,  Prompt_out), dim=1)                              
                x = torch.cat((x,  Prompt_in), dim=1)
                num_tokens = x.shape[1]
                x_ = self.blocks[i](x, temp=temp)
                x = x_[:, :num_tokens - (Prompt_Token_num)]
                Prompt_out_end = x_[:, num_tokens - Prompt_Token_num:]
                if self.config.merge_prompt:
                    Prompt_out = x_[:, num_tokens - (Prompt_Token_num):].flatten(1)
                    Prompt_out = self.tokens_proj(Prompt_out).reshape(x.shape[0],1,self.Prompt_Tokens.shape[2]) 

                    
        elif self.VPT_type == "Shallow":
            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.reshape(-1,x.shape[2])
            Prompt_Tokens = Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            # Sequntially procees
            x = self.blocks(x)
        elif self.VPT_type == "insert":
            Prompt_Tokens = self.Prompt_Tokens.reshape(-1,x.shape[2])
            Prompt_Tokens = Prompt_Tokens.expand(x.shape[0], -1, -1)
            Prompt_Token_num = self.Prompt_Tokens.shape[1]
            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                if i == 0:
                    x = torch.cat((x, Prompt_Tokens), dim=1)  
                    num_tokens = x.shape[1]
                    x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]            
                else:
                    x = self.blocks[i](x)     
        
        x = self.norm(x)
                            
        #return self.pre_logits(x[:, 0])  # use cls token for cls head  
        #0901 test, prompt also used for cls
        x_pr = (x[:, 0]+Prompt_out_end.mean(dim=1))/2  #Prompt_out.squeeze(1)
        return self.pre_logits(x_pr)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        #x = self.prompt_learner(x)
        x = self.classifier(x)
        return x
