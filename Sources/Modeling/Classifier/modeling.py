from abc import ABC
from abc import abstractmethod


class ModelFactory(ABC):
    def __init__(self, file):
        self.file = file
        pass

    @abstractmethod
    def get_emb(self):
        pass

    @abstractmethod
    def load_from_file(self):
        pass

    @abstractmethod
    def save_to_file(self):
        pass


class FactoryLR(ModelFactory):
    def __init__(self, file):
        super(FactoryLR, self).__init__(file)

    def get_emb(self):
        return ProductLR()

    def load_from_file(self):
        print(f'load from {self.file}')

    def save_to_file(self, file):
        print(f'save to {file}')


class FactoryMLP(ModelFactory):
    def __init__(self, file):
        super(FactoryMLP, self).__init__(file)

    def get_emb(self):
        return ProductMLP()

    def load_from_file(self):
        print(f'load from {self.file}')

    def save_to_file(self, file):
        print(f'save to {file}')


class FacotryRF(ModelFactory):
    def __init__(self, file):
        super(FacotryRF, self).__init__(file)

    def get_emb(self):
        return ProductRF()

    def load_from_file(self):
        print(f'load from {self.file}')

    def save_to_file(self, file):
        print(f'save to {file}')


class FactoryNN(ModelFactory):
    def __init__(self, file):
        super(FactoryNN, self).__init__(file)

    def get_emb(self):
        return ProductNN()

    def load_from_file(self):
        print(f'load from {self.file}')

    def save_to_file(self, file):
        print(f'save to {file}')


class FactorySVM(ModelFactory):
    def __init__(self, file):
        super(FactorySVM, self).__init__(file)

    def get_emb(self):
        return ProductSVM()

    def load_from_file(self):
        print(f'load from {self.file}')

    def save_to_file(self, file):
        print(f'save to {file}')

#=====================
#==Products
#=====================

class ModelProduct(ABC):
    @abstractmethod
    def run(self):
        pass


class ProductRF(ModelProduct):

    def run(self):
        pass


class ProductMLP(ModelProduct):

    def run(self):
        pass


class ProductSVM(ModelProduct):

    def run(self):
        pass


class ProductNN(ModelProduct):

    def run(self):
        pass


class ProductLR(ModelProduct):

    def run(self):
        pass


if __name__ == '__main__':
    pass
    # node2vecA = FactoryNode2vec("fileA")
    # print(node2vecA.get_emb().run())
    # node2vecA.load_from_file()
    # node2vecA.save_to_file("save to fileC")
    #
    # node2vecB = FactoryNode2vec("fileB")
    # print(node2vecB.get_emb().run())
    # node2vecB.load_from_file()
    # node2vecB.save_to_file("save to fileD")
