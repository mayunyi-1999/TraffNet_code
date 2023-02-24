import torch

model = torch.load('pathSelParams_sumosy/pathSel_sumosy.pkl')
torch.save(model.state_dict(),'pathSelParams_sumosy.pth')


model = torch.load('pathSelParams_taxibj/pathSel_taxibj.pkl')
torch.save(model.state_dict(),'pathSelParams_taxibj.pth')

model = torch.load('pathSelParams_sumovs/pathSel_sumovs.pkl')
torch.save(model.state_dict(),'pathSelParams_sumovs.pth')


print(model)