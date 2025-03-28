import torch
from model import make_model
from masking import subsequent_mask

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)      
    ys = torch.zeros(1, 1).type_as(src.data)  # start token

    for i in range(9):
        out = test_model.decode( memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src) )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)  # get index of max prob
        next_word = next_word.data[0]        
        ys = torch.cat( [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1 )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()
        
        
if __name__ == "__main__":
    run_tests()
    
    # Expected output:
    #    
    # Example Untrained Model Prediction: tensor([[0, 4, 4, 4, 2, 4, 4, 4, 4, 4]])
    # Example Untrained Model Prediction: tensor([[ 0, 10,  1,  0,  0, 10,  1,  0, 10,  1]])
    # Example Untrained Model Prediction: tensor([[ 0,  6,  8,  7,  6,  2,  4,  2,  4, 10]])
    # Example Untrained Model Prediction: tensor([[0, 4, 2, 2, 9, 7, 9, 6, 3, 7]])
    # Example Untrained Model Prediction: tensor([[0, 7, 7, 7, 7, 7, 7, 7, 7, 7]])
    # Example Untrained Model Prediction: tensor([[ 0, 10,  4,  0, 10,  4,  0, 10,  4,  0]])
    # Example Untrained Model Prediction: tensor([[ 0,  6, 10,  6, 10,  5, 10,  6, 10,  6]])
    # Example Untrained Model Prediction: tensor([[0, 3, 0, 3, 7, 0, 3, 3, 3, 0]])
    # Example Untrained Model Prediction: tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # Example Untrained Model Prediction: tensor([[0, 2, 6, 3, 3, 3, 3, 3, 3, 3]])