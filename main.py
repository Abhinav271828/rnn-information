from utils import *

classes = [('nonlin', ToyRNN),
           ('tanh', ToyRNN),
          ]

descriptions = []
for is_ in [7, 8]:
    for hs in range(10,3,-1):
        for type in ['nonlin', 'tanh']:
            for bias in [True, False]:
                descriptions.append(f'in={is_}-out={hs}-{type}-bias={bias}-offset=0')

def load_model_from_index(index):
    path = descriptions[index] + '.ckpt'
    model_type = dict(classes)[path.split('-')[2]]
    mdl = load_model_from_name(model_type, path)
    return mdl

def create_model_from_index(index):
    path = descriptions[index]
    is_ = int(path.split('-')[0].split('=')[1])
    hs = int(path.split('-')[1].split('=')[1])
    bias = eval(path.split('-')[3].split('=')[1])
    offset = int(path.split('-')[4].split('=')[1])
    model_type = path.split('-')[2]
    cls = dict(classes)[model_type]
    if model_type == 'tanh':
        mdl = cls(input_size=is_, hidden_size=hs, bias=bias, offset=offset, nonlinearity='tanh')
    else:
        mdl = cls(input_size=is_, hidden_size=hs, bias=bias, offset=offset)
    
    train_and_save(mdl, descriptions[index])
    #check_dimensionality(mdl)
    #sv_box_plot(mdl)
    return mdl

if __name__ == '__main__':
    lbs = []
    ubs = []
    x = OffsetData(7, 0, 1000, 1)[0][0]

    for e in []:
        path = descriptions[e]
        print(f"Training model {descriptions[e]}")
        create_model_from_index(e)
        print("Training complete!")

    x = OffsetData(8, 0, 1000, 1)[0][0]
    for e in [28, 32, 33, 37, 40, 41] + list(range(42, 56)):
        path = descriptions[e]
        print(f"Training model {descriptions[e]}")
        create_model_from_index(e)
        print("Training complete!")

    line = (go.Figure().add_trace(go.Scatter(x=list(range(len(descriptions))), y=lbs))
                       .add_trace(go.Scatter(x=list(range(len(descriptions))), y=ubs)))
    line.show()
