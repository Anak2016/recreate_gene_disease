from abc import ABC
from abc import abstractmethod
from node2vec import Node2Vec


class EmbFactory(ABC):
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



class FactoryNode2vec(EmbFactory):
    def __init__(self, file):
        super(FactoryNode2vec, self).__init__(file)

    def get_emb(self):
        return ProductNode2Vec()

    def load_from_file(self):
        print(f'load from {self.file}')

    def save_to_file(self, file):
        print(f'save to {file}')

# class FactoryBine(EmbFactory):
#     def __init__(self, file):
#         super(FactoryBine, self).__init__(file)
#
#     def get_emb(self):
#         return ProductBINE()
#
#     def load_from_file(self):
#         print(f'load from {self.file}')
#
#     def save_to_file(self, file):
#         print(f'save to {file}')



class EmbProduct(ABC):
    @abstractmethod
    def run(self):
        pass



class ProductNode2Vec(EmbProduct):

    def run(self):
        print('in productnode2vec.run()')
        node2vec = node2vec(g, dimensions=64, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embedding_model_filename = r'c:\users\anak\pycharmprojects\recreate_gene_disease\data\processed\genediseaseproject\copd\node2vec'
        model.save(embedding_model_filename)
        # return 'run'


# class ProductBINE(EmbProduct):
#
#     def run(self):
#         import bine
#         bine.run()

if __name__ == '__main__':

    node2vecA = FactoryNode2vec("fileA")
    print(node2vecA.get_emb().run())
    node2vecA.load_from_file()
    node2vecA.save_to_file("save to fileC")

    # node2vecB = FactoryNode2vec("fileB")
    # print(node2vecB.get_emb().run())
    # node2vecB.load_from_file()
    # node2vecB.save_to_file("save to fileD")
