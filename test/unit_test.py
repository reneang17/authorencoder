
import unittest
import os
import sys
import pickle
sys.path.insert(1, '../src/')
sys.path.insert(1, '../process/')
sys.path.insert(1, '../wrangling_eda/')

from models import CNN
import torch

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        pass

    # Returns True if the string contains 4 a.
    def test_strings_a(self):
        #tensor = torch.zeros((1, 13, 17), dtype=torch.int32)
        #self.assertEqual(tensor.size() ,torch.Size([1, 13, 17]) )
        INPUT_DIM = 20_100
        WORD_EMBEDDING_DIM = 100  #Fixed by preloaded embedding
        N_FILTERS = 100
        FILTER_SIZES = [2,3,4]
        AUTHOR_DIM = 2
        DROPOUT = 0.5
        PAD_IDX = 1
        tensor = torch.zeros((13, 17), dtype=torch.int32).unsqueeze(0).to(torch.int64)
        model = CNN(INPUT_DIM, WORD_EMBEDDING_DIM, N_FILTERS, FILTER_SIZES,
        AUTHOR_DIM, DROPOUT, PAD_IDX)
        self.assertEqual(model(tensor).size() , torch.Size([13, 2]) )


    # Returns True if the string is in upper case.
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    # Returns TRUE if the string is in uppercase
    # else returns False.
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    # Returns true if the string is stripped and
    # matches the given output.
    def test_strip(self):
        s = 'geeksforgeeks'
        self.assertEqual(s.strip('geek'), 'sforgeeks')

    # Returns true if the string splits and matches
    # the given output.
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
