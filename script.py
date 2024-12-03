from FlagEmbedding import FlagAutoModel
import torch



if __name__ == '__main__':
    # device = torch.device('cuda:1')
    model = FlagAutoModel.from_finetuned('/home/models/bge-m3',
                                         query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                         use_fp16=True,
                                         device="cuda:1")
    # model.to(device)
    sentences_1 = ["I love NLP", "I love machine learning"]
    sentences_2 = ["I love BGE", "I love text retrieval"]
    embeddings_1 = model.encode(sentences_1)
    embeddings_2 = model.encode(sentences_2)
    print(type(embeddings_2))