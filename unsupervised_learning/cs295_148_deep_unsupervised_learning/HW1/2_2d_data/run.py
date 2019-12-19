from src.model_1 import Model1
from src.model_2 import Model2
from src.training import train_model


if __name__ == "__main__":

    model1 = Model1()
    model2 = Model2()

    train_model(model1)
    train_model(model2)
