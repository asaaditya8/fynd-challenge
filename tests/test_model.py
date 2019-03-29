from src.models import model
from keras.models import Model

def test_create_model():
    assert type(model.create_model(weights=None)) == Model