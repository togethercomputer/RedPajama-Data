import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cc_net.roots_func import filter_small_doc, filter_small_docs_by_bytes

class TestFilterSmallDoc(unittest.TestCase):
    def test_filter_small_doc(self):
        doc = {
            'text': 'Super Bowl\n\nChoose Su\nper Bowl yooooooo\nSearch \nDatabase Home my favorate\n'
        }
        expected = {
            'text': 'per Bowl yooooooo\nDatabase Home my favorate\n'   
        }
        result = filter_small_doc('text').do(doc)
        self.assertEqual(result, expected)

    def test_filter_small_doc_empty(self):
        doc = {}
        expected = None
        result = filter_small_doc('text').do(doc)
        self.assertEqual(result, expected)

    def test_filter_small_docs_by_bytes(self):
        doc1 = {
            'text': 'Super Bowl\nChoose Su\nper Bowl \nSearch \nDat\nabase Home'
        }
        doc2 = {
            'text': 'a' * 1000
        }
        result1 = filter_small_docs_by_bytes('text').do(doc1)
        result2 = filter_small_docs_by_bytes('text').do(doc2)
        self.assertEqual(result1, None)
        self.assertEqual(result2, doc2)  

    def test_filter_small_docs_by_bytes_empty(self):
        doc = {}
        expected = None
        result = filter_small_docs_by_bytes('text').do(doc)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()