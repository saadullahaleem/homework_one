import torch
import config
from model import BertBaseUncased
from flask import Flask
from flask_restful import Resource, Api


def prediction(text):
    encoded = config.TOKENIZER.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        pad_to_max_length=True,
        return_attention_mask=True
    )
    ids = torch.tensor(encoded['input_ids'],       dtype=torch.long).unsqueeze(0)
    masks = torch.tensor(encoded['attention_mask'], dtype=torch.long).unsqueeze(0)
    t_id = torch.tensor(encoded['token_type_ids'], dtype=torch.long).unsqueeze(0)
    ids = ids.to(device, dtype=torch.long)
    masks = masks.to(device, dtype=torch.long)
    t_id = t_id.to(device, dtype=torch.long)
    with torch.no_grad():
        output = model(
            ids=ids,
            masks=masks,
            token_type_ids=t_id
        )
        return torch.sigmoid(output).cpu().detach().numpy()

app = Flask(__name__)
api = Api(app)


class PredictorView(Resource):
    """
    This view will eventually integrate the model.
    For now it sends sample json data
    """
    def get(self):
        return {
            'William Shakespeare': {
                'quote': ['Love all,trust a few,do wrong to none',
                'Some are born great, some achieve greatness, and some greatness thrust upon them.']
        },
        'Linus': {
            'quote': ['Talk is cheap. Show me the code.']
            }
        }

    @app.get("/{sentence}")
    def get(self, sentence: str):
        pred = prediction(sentence)[0][0]
        return {"message": str(pred)}


api.add_resource(PredictorView, '/predict')

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = 'model.bin'
    model = BertBaseUncased()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    app.run(host='0.0.0.0', port="5000", debug=True)